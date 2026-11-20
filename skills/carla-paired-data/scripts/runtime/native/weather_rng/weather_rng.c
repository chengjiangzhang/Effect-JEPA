#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <gnu/libc-version.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

/* ABI pinned to Shipping SHA 03bcd413615fa1fc61a5b846342dbdc4e6b3541320a40d6bf17ff927039731f9.
 * Only BP_Weather_C.Rain's verified RandomFloatInRange result is replaced.
 * Every actual libc call is still consumed exactly once; other callers retain it.
 * Private state is initialized once per native process, never per actor/map/frame.
 */
#define POLICY "isolated_bp_weather_rain_v1"
#define TARGET_CALLER ((uintptr_t)0x5ab3d5a)
#define NATIVE_NAME_POOL ((uintptr_t)0x740d300)
static int (*libc_rand)(void);
static void (*libc_srand)(unsigned);
static _Thread_local int resolving;
static int enabled, output_fd=-1;
static unsigned private_seed;
static struct random_data private_data;
static _Alignas(16) char private_state[128];
static unsigned long long weather_calls;
static _Atomic unsigned long long logged_events;
static pthread_mutex_t weather_mutex=PTHREAD_MUTEX_INITIALIZER;

static void resolve(void) {
    if (libc_rand && libc_srand) return;
    if (resolving) _exit(126);
    resolving=1;libc_rand=(int(*)(void))dlsym(RTLD_NEXT,"rand");
    libc_srand=(void(*)(unsigned))dlsym(RTLD_NEXT,"srand");
    if (!libc_rand || !libc_srand) _exit(126);
    resolving=0;
}

static int peek(uintptr_t address, void *out, size_t bytes) {
    if (!address || bytes > 256) return 0;
    struct iovec local = {out, bytes}, remote = {(void *)address, bytes};
    return syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1, 0) == (ssize_t)bytes;
}
static int read_ptr(uintptr_t address, uintptr_t *out) {
    return peek(address, out, sizeof(*out)) && *out >= 4096 && !(*out & 7);
}
static int object_name(uintptr_t object, char out[128], uintptr_t pool_blocks) {
    uint32_t index; uintptr_t block; uint16_t header;
    out[0] = 0;
    if (!peek(object+24, &index, 4)) return 0;
    if (!read_ptr(pool_blocks + ((index >> 16) & 0xffffu)*8, &block)) return 0;
    uintptr_t entry = block + (index & 0xffffu)*2;
    if (!peek(entry, &header, 2)) return 0;
    unsigned length = header >> 6;
    if (!length || length > 127) return 0;
    if (header & 1) {
        uint16_t wide[128];
        if (!peek(entry+2, wide, length*2)) return 0;
        for (unsigned i=0; i<length; ++i) out[i] = wide[i] >= 32 && wide[i] <= 126 && wide[i] != '"' && wide[i] != '\\' ? (char)wide[i] : '?';
    } else {
        if (!peek(entry+2, out, length)) return 0;
        for (unsigned i=0; i<length; ++i) if ((unsigned char)out[i]<32 || (unsigned char)out[i]>126 || out[i]=='"' || out[i]=='\\') out[i]='?';
    }
    out[length]=0; return 1;
}

struct Context {
    int valid, rain_identity, matched;
    unsigned script_size;
    unsigned long long code_offset;
    char node[128], outer[128], object[128], klass[128];
};
static struct Context read_context(uintptr_t frame, uintptr_t pool) {
    struct Context c={0};uintptr_t node=0,object=0,code=0,outer=0,klass=0,script=0;
    if (!read_ptr(frame+16,&node) || !read_ptr(frame+24,&object) || !peek(frame+32,&code,8)) return c;
    if (!object_name(node,c.node,pool) || !object_name(object,c.object,pool)) return c;
    if (!read_ptr(node+32,&outer) || !object_name(outer,c.outer,pool)) return c;
    if (!read_ptr(object+16,&klass) || !object_name(klass,c.klass,pool)) return c;
    c.rain_identity=!strcmp(c.node,"Rain") && !strcmp(c.outer,"BP_Weather_C") && !strcmp(c.klass,"BP_Weather_C");
    if (!read_ptr(node+96,&script) || !peek(node+104,&c.script_size,4) || c.script_size>1048576 || code<script || code>script+c.script_size) return c;
    c.valid=1;c.code_offset=code-script;
    c.matched=c.rain_identity && c.script_size==3565 && (c.code_offset==1584 || c.code_offset==3304);
    return c;
}
static void write_line(const char *line,int n) {
    if (output_fd<0 || n<0) return;
    if (write(output_fd,line,(size_t)n)!=n) _exit(125);
}
static void log_draw(int actual,int result,const struct Context *c,unsigned long long ordinal,const char *status) {
    if (output_fd<0) return;
    unsigned long long index=atomic_fetch_add(&logged_events,1)+1;
    if(index>1024) return;
    char line[1536];int n=snprintf(line,sizeof(line),
        "{\"kind\":\"weather_draw\",\"status\":\"%s\",\"event_index\":%llu,\"tid\":%ld,\"actual_libc_value\":%d,\"returned_value\":%d,\"weather_ordinal\":%llu,\"context_valid\":%s,\"weather_match\":%s,\"node\":\"%s\",\"node_outer\":\"%s\",\"object_class\":\"%s\",\"code_offset\":%llu,\"script_size\":%u}\n",
        status,index,syscall(SYS_gettid),actual,result,ordinal,c->valid?"true":"false",c->matched?"true":"false",c->node,c->outer,c->klass,c->code_offset,c->script_size);
    if(n<=0 || (size_t)n>=sizeof(line)) _exit(125);
    write_line(line,n);
}
static void initialize_private(unsigned seed) {
    memset(&private_data,0,sizeof(private_data));memset(private_state,0,sizeof(private_state));
    if (initstate_r(seed,private_state,sizeof(private_state),&private_data)!=0 || private_data.rand_type!=3 || private_data.rand_deg!=31 || private_data.rand_sep!=3) _exit(125);
    private_seed=seed;weather_calls=0;
}
/* Unit tests pass a synthetic readable pool; native code always uses the pinned pool. */
static int select_value(int actual,uintptr_t caller,uintptr_t frame,uintptr_t pool) {
    if (!enabled || caller!=TARGET_CALLER) return actual;
    struct Context c=read_context(frame,pool);
    if(c.rain_identity && (!c.valid || !c.matched)) {
        log_draw(actual,actual,&c,0,"unsupported_weather_script");_exit(125);
    }
    if (!c.valid || !c.matched) {log_draw(actual,actual,&c,0,"unchanged_non_weather");return actual;}
    if (syscall(SYS_gettid)!=getpid()) {
        log_draw(actual,actual,&c,0,"unsupported_weather_thread");_exit(125);
    }
    if(pthread_mutex_lock(&weather_mutex)!=0) _exit(125);
    int32_t value;
    if(random_r(&private_data,&value)!=0) _exit(125);
    unsigned long long ordinal=++weather_calls;
    if(pthread_mutex_unlock(&weather_mutex)!=0) _exit(125);
    log_draw(actual,value,&c,ordinal,"isolated_weather_value");return value;
}
/* Deterministic tie ordering after the original visible-decal builder.
 * Preserve culling, explicit SortOrder, normal/blend categories, materials,
 * transforms, fade, rendering settings and all original draw records.
 * Only pointer-derived material/component ties are replaced by object paths.
 */
#include <sys/mman.h>
static int decal_order_fd=-1;
static unsigned long long decal_order_calls,decal_order_changed;
static void decal_fatal_at(int line) {
    char message[128];int n=snprintf(message,sizeof(message),"{\"kind\":\"decal_order_error\",\"source_line\":%d}\n",line);
    if(decal_order_fd>=0 && n>0 && (size_t)n<sizeof(message)) {ssize_t ignored=write(decal_order_fd,message,(size_t)n);(void)ignored;}
    _exit(122);
}
#define DECAL_FAIL() decal_fatal_at(__LINE__)

static unsigned char (*original_decal_list)(uintptr_t,uintptr_t,int,uintptr_t);
static int exact_hook_n(uintptr_t address,const unsigned char *expected,size_t length,
                         void *replacement,void **original) {
    unsigned char found[64],jump[14]={0xff,0x25,0,0,0,0};
    if(length<14 || length>sizeof(found) || !peek(address,found,length) || memcmp(found,expected,length)) return 0;
    unsigned char *code=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(code==MAP_FAILED)return 0;
    uintptr_t continuation=address+length;
    memcpy(code,expected,length);memcpy(jump+6,&continuation,8);memcpy(code+length,jump,14);
    if(mprotect(code,4096,PROT_READ|PROT_EXEC))return 0;
    uintptr_t page=address&~(uintptr_t)4095;
    size_t span=(address+length-page+4095)&~(size_t)4095;
    if(mprotect((void *)page,span,PROT_READ|PROT_WRITE|PROT_EXEC))return 0;
    *original=code;memcpy(jump+6,&replacement,8);memcpy((void *)address,jump,14);
    memset((void *)(address+14),0x90,length-14);
    __builtin___clear_cache((char *)address,(char *)(address+length));
    return mprotect((void *)page,span,PROT_READ|PROT_EXEC)==0;
}
static int decal_stable_path(uintptr_t object,char out[1536]) {
    char names[10][128];uint32_t numbers[10];unsigned n=0;uintptr_t p=object;
    while(p && n<10) {
        if(!object_name(p,names[n],NATIVE_NAME_POOL) || !peek(p+28,&numbers[n],4) || !peek(p+32,&p,8))return 0;
        /* This pinned asset set uses ASCII paths. Never silently collapse a
         * sanitized or truncated identity to a common tie key. */
        if(strchr(names[n],'?'))return 0;
        ++n;
    }
    if(p || !n)return 0;
    size_t used=0;out[0]=0;
    for(unsigned i=n;i>0;--i) {
        int k=snprintf(out+used,1536-used,"%s%s#%u",used?"/":"",names[i-1],numbers[i-1]);
        if(k<0 || (size_t)k>=1536-used)return 0;
        used+=(size_t)k;
    }
    return 1;
}
struct DecalStableKey {char material[1536],component[1536];int32_t sort,blend;unsigned char normal;};
/* Capture UObject identities during game-thread proxy construction. They must
 * not be dereferenced later from a queued render pass while a map unloads. */
#define DECAL_CACHE_CAP 16384u
struct DecalCachedIdentity {uintptr_t proxy,component,material;char material_path[1536],component_path[1536];};
static struct DecalCachedIdentity *decal_cache;
static pthread_mutex_t decal_cache_lock=PTHREAD_MUTEX_INITIALIZER;
static void (*original_decal_construct)(uintptr_t,uintptr_t);
static unsigned cache_bucket(uintptr_t proxy) {
    uint64_t h=(uint64_t)(proxy>>4);h^=h>>33;h*=UINT64_C(0xff51afd7ed558ccd);h^=h>>33;
    return (unsigned)h&(DECAL_CACHE_CAP-1);
}
static struct DecalCachedIdentity *cache_find(uintptr_t proxy,int create) {
    unsigned b=cache_bucket(proxy);
    for(unsigned i=0;i<DECAL_CACHE_CAP;++i) {
        struct DecalCachedIdentity *entry=&decal_cache[(b+i)&(DECAL_CACHE_CAP-1)];
        if(entry->proxy==proxy || (!entry->proxy && create))return entry;
        if(!entry->proxy)return NULL;
    }
    return NULL;
}
static void decal_construct_wrapper(uintptr_t proxy,uintptr_t input_component) {
    original_decal_construct(proxy,input_component);int saved=errno;
    uintptr_t component=0,material=0;
    char component_path[1536],material_path[1536];
    if(!read_ptr(proxy,&component) || component!=input_component || !read_ptr(proxy+8,&material)
       || !decal_stable_path(component,component_path) || !decal_stable_path(material,material_path))DECAL_FAIL();
    if(pthread_mutex_lock(&decal_cache_lock))DECAL_FAIL();
    struct DecalCachedIdentity *entry=cache_find(proxy,1);if(!entry)DECAL_FAIL();
    entry->proxy=proxy;entry->component=component;entry->material=material;
    strcpy(entry->component_path,component_path);strcpy(entry->material_path,material_path);
    if(pthread_mutex_unlock(&decal_cache_lock))DECAL_FAIL();
    errno=saved;
}
static int cached_decal_paths(uintptr_t proxy,uintptr_t component,uintptr_t material,struct DecalStableKey *key) {
    if(pthread_mutex_lock(&decal_cache_lock))DECAL_FAIL();
    struct DecalCachedIdentity *entry=cache_find(proxy,0);
    int valid=entry && entry->component==component && entry->material==material;
    if(valid) {strcpy(key->component,entry->component_path);strcpy(key->material,entry->material_path);}
    if(pthread_mutex_unlock(&decal_cache_lock))DECAL_FAIL();
    return valid;
}

struct DecalStableRow {unsigned char original[40];struct DecalStableKey *key;unsigned index;};
static int compare_decal_stable(const void *pa,const void *pb) {
    const struct DecalStableKey *a=((const struct DecalStableRow *)pa)->key,*b=((const struct DecalStableRow *)pb)->key;
    if(a->sort!=b->sort)return a->sort<b->sort?-1:1;
    if(a->normal!=b->normal)return a->normal>b->normal?-1:1;
    if(a->blend!=b->blend)return a->blend>b->blend?-1:1;
    int c=strcmp(a->material,b->material);
    return c?c:strcmp(a->component,b->component);
}
static uint64_t stable_hash(uint64_t h,const void *p,size_t n) {
    const unsigned char *b=p;for(size_t i=0;i<n;++i){h^=b[i];h*=UINT64_C(1099511628211);}return h;
}
static void decal_order_line(const char *line,int n,size_t cap) {
    if(decal_order_fd<0 || n<=0 || (size_t)n>=cap || write(decal_order_fd,line,(size_t)n)!=n)DECAL_FAIL();
}
static unsigned char decal_list_wrapper(uintptr_t scene,uintptr_t view,int stage,uintptr_t array) {
    unsigned char result=original_decal_list(scene,view,stage,array);
    int saved_errno=errno;
    if(!array){errno=saved_errno;return result;}
    uintptr_t data=0;int32_t count=0,max=0;
    if(!peek(array,&data,8) || !peek(array+8,&count,4) || !peek(array+12,&max,4)
       || count<0 || count>65536 || max<count || (count && !data))DECAL_FAIL();
    if(count<2){errno=saved_errno;return result;}
    struct DecalStableKey *keys=calloc((size_t)count,sizeof(*keys));
    struct DecalStableRow *rows=calloc((size_t)count,sizeof(*rows));
    if(!keys || !rows)DECAL_FAIL();
    for(int32_t i=0;i<count;++i) {
        uintptr_t proxy=0,component=0,material=0;
        if(!peek(data+(size_t)i*40,rows[i].original,40))DECAL_FAIL();
        memcpy(&proxy,rows[i].original+16,8);
        if(!read_ptr(proxy,&component) || !read_ptr(proxy+8,&material)
           || !peek(proxy+68,&keys[i].sort,4)
           || !cached_decal_paths(proxy,component,material,&keys[i]))DECAL_FAIL();
        memcpy(&keys[i].blend,rows[i].original+32,4);keys[i].normal=rows[i].original[36];
        rows[i].key=&keys[i];rows[i].index=(unsigned)i;
    }
    qsort(rows,(size_t)count,sizeof(*rows),compare_decal_stable);
    uint64_t h=UINT64_C(14695981039346656037);unsigned moved=0;
    for(int32_t i=0;i<count;++i) {
        /* Distinct objects must not collapse to an unresolved path tie. */
        if(i && !compare_decal_stable(&rows[i-1],&rows[i])
           && memcmp(rows[i-1].original+16,rows[i].original+16,8))DECAL_FAIL();
        moved+=rows[i].index!=(unsigned)i;
        memcpy((void *)(data+(size_t)i*40),rows[i].original,40);
        struct DecalStableKey *k=rows[i].key;
        h=stable_hash(h,k->material,strlen(k->material)+1);h=stable_hash(h,k->component,strlen(k->component)+1);
        h=stable_hash(h,&k->sort,4);h=stable_hash(h,rows[i].original+24,13);
    }
    free(rows);free(keys);
    uintptr_t actor=0,family=0;uint32_t actor_number=0,frame=0;float time=0;char actor_name[128]="";
    if(!peek(view+576,&actor,8) || !read_ptr(view,&family) || !peek(family+80,&time,4) || !peek(family+92,&frame,4))DECAL_FAIL();
    if(actor && (!object_name(actor,actor_name,NATIVE_NAME_POOL) || !peek(actor+28,&actor_number,4)))DECAL_FAIL();
    ++decal_order_calls;decal_order_changed+=moved>0;
    if(decal_order_calls>100000)DECAL_FAIL();
    char line[512];int n=snprintf(line,sizeof(line),
        "{\"kind\":\"decal_order_apply\",\"ordinal\":%llu,\"actor\":\"%s\",\"actor_number\":%u,\"frame\":%u,\"time\":%.9g,\"stage\":%d,\"count\":%d,\"moved\":%u,\"ordered_input_hash\":\"%016llx\"}\n",
        decal_order_calls,actor_name,actor_number,frame,time,stage,count,moved,(unsigned long long)h);
    decal_order_line(line,n,sizeof(line));errno=saved_errno;return result;
}
static void start_decal_order(const char *directory) {
    char path[PATH_MAX];int n=snprintf(path,sizeof(path),"%s/decal-order-%ld.jsonl",directory,(long)getpid());
    if(n<=0 || (size_t)n>=sizeof(path))DECAL_FAIL();
    decal_order_fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_APPEND,0644);if(decal_order_fd<0)DECAL_FAIL();
    decal_cache=calloc(DECAL_CACHE_CAP,sizeof(*decal_cache));if(!decal_cache)DECAL_FAIL();
    const unsigned char constructor_entry[14]={0x41,0x57,0x41,0x56,0x41,0x54,0x53,0x50,0x48,0x89,0xf3,0x49,0x89,0xfe};
    if(!exact_hook_n(0x525dcd0,constructor_entry,14,(void *)decal_construct_wrapper,(void **)&original_decal_construct))DECAL_FAIL();
    const unsigned char expected[17]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0x08,0x04,0,0};
    if(!exact_hook_n(0x3bf64f0,expected,sizeof(expected),(void *)decal_list_wrapper,(void **)&original_decal_list))DECAL_FAIL();
    const char *header="{\"kind\":\"decal_order_start\",\"policy\":\"stable_decal_paths_v2\",\"target\":\"0x3bf64f0\",\"entry_verified\":true,\"preserved_primary_order\":[\"SortOrder_ascending\",\"bHasNormal_descending\",\"FinalDecalBlendMode_descending\"],\"stable_ties\":[\"material_object_path\",\"component_object_path\"]}\n";
    decal_order_line(header,(int)strlen(header),strlen(header)+1);
}
static void finish_decal_order(void) {
    if(decal_order_fd<0)return;
    char line[256];int n=snprintf(line,sizeof(line),"{\"kind\":\"decal_order_end\",\"calls\":%llu,\"changed_lists\":%llu}\n",decal_order_calls,decal_order_changed);
    decal_order_line(line,n,sizeof(line));close(decal_order_fd);
}

__attribute__((visibility("hidden"))) int weather_rng_rand(uintptr_t caller,uintptr_t frame) {
    if(!libc_rand) resolve();
    int actual=libc_rand();int saved_errno=errno;
    int result=select_value(actual,caller,frame,NATIVE_NAME_POOL);
    errno=saved_errno;return result;
}
void srand(unsigned seed) {
    if(!libc_srand) resolve();
    libc_srand(seed);int saved_errno=errno;
    if(output_fd>=0) {
        char line[192];int n=snprintf(line,sizeof(line),"{\"kind\":\"actual_libc_srand\",\"seed\":%u,\"private_weather_stream_reset\":false}\n",seed);
        if(n>0 && (size_t)n<sizeof(line)) write_line(line,n);
    }
    errno=saved_errno;
}
__attribute__((constructor)) static void start(void) {
    int saved_errno=errno;resolve();
    char executable[PATH_MAX];ssize_t n=readlink("/proc/self/exe",executable,sizeof(executable)-1);
    if(n<0){errno=saved_errno;return;}executable[n]=0;
    const char *name=strrchr(executable,'/');name=name?name+1:executable;
    int test=!strcmp(name,"weather_rng_probe") && getenv("CARLA_WEATHER_RNG_ALLOW_TEST") && !strcmp(getenv("CARLA_WEATHER_RNG_ALLOW_TEST"),"1");
    if(strcmp(name,"CarlaUE4-Linux-Shipping") && !test){errno=saved_errno;return;}
    const char *seed=getenv("CARLA_WEATHER_RNG_SEED"),*directory=getenv("CARLA_WEATHER_RNG_LOG_DIR");
    if(!seed || !directory || directory[0]!='/') _exit(126);
    char *end;errno=0;unsigned long parsed=strtoul(seed,&end,10);
    if(errno || *end || end==seed || seed[0]=='-' || parsed>UINT_MAX) _exit(126);
    initialize_private((unsigned)parsed);
    char path[PATH_MAX];n=snprintf(path,sizeof(path),"%s/weather-%ld.jsonl",directory,(long)getpid());
    if(n<=0 || (size_t)n>=sizeof(path)) _exit(126);
    output_fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_APPEND,0644);
    if(output_fd<0) _exit(126);
    char line[640];n=snprintf(line,sizeof(line),
        "{\"kind\":\"weather_rng_start\",\"policy\":\"%s\",\"pid\":%ld,\"seed\":%u,\"algorithm\":\"glibc_random_r_type3_deg31_sep3\",\"state_bytes\":128,\"libc_version\":\"%s\",\"libc_consumption_preserved\":true,\"target_caller\":\"0x5ab3d5a\",\"main_thread_only\":true,\"test_probe\":%s}\n",
        POLICY,(long)getpid(),private_seed,gnu_get_libc_version(),test?"true":"false");
    if(n<=0 || (size_t)n>=sizeof(line)) _exit(126);
    write_line(line,(int)n);
    start_decal_order(directory);
    enabled=1;errno=saved_errno;
}
__attribute__((destructor)) static void finish(void) {
    if(output_fd<0)return;
    char line[192];int n=snprintf(line,sizeof(line),"{\"kind\":\"weather_rng_end\",\"weather_calls\":%llu,\"selected_events\":%llu,\"truncated\":%s}\n",weather_calls,atomic_load(&logged_events),atomic_load(&logged_events)>1024?"true":"false");
    if(n>0 && (size_t)n<sizeof(line))write_line(line,n);
    close(output_fd);
    finish_decal_order();
}
