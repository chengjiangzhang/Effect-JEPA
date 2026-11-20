.text
.globl rand
.type rand,@function
rand:
    mov %rbx, %rsi
    mov (%rsp), %rdi
    jmp weather_rng_rand
.size rand, .-rand
.section .note.GNU-stack,"",@progbits
