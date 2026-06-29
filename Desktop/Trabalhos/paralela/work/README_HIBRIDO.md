# Ray Tracer Híbrido MPI + OpenMP

Versão híbrida do ray tracer para execução em 4 nós do cluster Atlantica.

## Arquitetura

- **MPI** (memória distribuída): 1 processo *pesado* por nó. O coordenador
  (rank 0) distribui **blocos de linhas** (chunks) sob demanda; os
  trabalhadores pedem trabalho conforme terminam (escalonamento dinâmico).
- **OpenMP** (memória compartilhada): dentro de cada nó, as linhas do chunk
  recebido são renderizadas por todas as threads em modelo **workpool**
  (`#pragma omp parallel for schedule(dynamic, 1)` — todas as threads
  trabalham, controle pela estrutura do OpenMP, sem mestre interno).

```
4 nós  ->  4 processos MPI (1 por nó)
 ├─ rank 0: coordenador (+ trabalhador, no modo híbrido)
 └─ rank 1..3: trabalhadores
        └─ cada processo: N threads OpenMP sobre as linhas do chunk
```

## Compilação

    make            # usa mpic++ com -fopenmp

## Variáveis de controle (ambiente)

| Variável        | Padrão  | Efeito                                                        |
|-----------------|---------|--------------------------------------------------------------|
| `OMP_NUM_THREADS` | (sistema) | Nº de threads OpenMP por nó. Use 16 no Atlantica (8c/16t). |
| `CHUNK_LINES`     | 16      | Linhas por bloco MPI. Alimenta as threads sem ociosidade.   |
| `COORD_MODE`      | worker  | `worker` = rank 0 coordena E renderiza; `dedicated` = só coordena. |

## Execução básica

    # Híbrido, 4 processos, 16 threads cada, coordenador trabalhando:
    OMP_NUM_THREADS=16 CHUNK_LINES=16 \
      mpirun --hostfile hosts.txt --map-by node --bind-to none -np 4 ./build/main 3000

    # Coordenador dedicado (para a análise comparativa):
    OMP_NUM_THREADS=16 COORD_MODE=dedicated \
      mpirun --hostfile hosts.txt --map-by node -np 4 ./build/main 3000

A imagem sai em stdout (formato PPM/P3) — redirecione para arquivo.
As métricas (tempo, linhas por worker) saem em stderr.

## Experimentos do enunciado

`run_atlantica.sh` automatiza os 4 itens de avaliação:
1. **Strong scaling** (imagem fixa, 1→4 nós) — speedup e eficiência.
2. **Weak scaling** (carga proporcional aos nós).
3. **Coordenador dedicado vs. trabalhador** (4 nós).
4. **Híbrido vs. MPI puro** (4 nós × 16 = 64 processos, 1 thread cada).

Edite `HOSTFILE`, `THREADS` e os tamanhos de imagem conforme o cluster, depois:

    HOSTFILE=hosts.txt THREADS=16 ./run_atlantica.sh

## Notas de implementação importantes

- **RNG thread-safe**: `random_double()` foi reescrito com `std::mt19937`
  `thread_local`. O `std::rand()` original NÃO é thread-safe e causaria
  race conditions e contenção sob OpenMP.
- **Coordenador híbrido**: usa `MPI_Iprobe` (não-bloqueante). Quando não há
  mensagem pendente, o rank 0 renderiza um chunk próprio; assim ele não fica
  ocioso esperando, mas também não atrasa as respostas aos trabalhadores.
- **Economia de RAM**: trabalhadores alocam apenas `CHUNK_LINES * largura`;
  só o coordenador aloca o buffer da imagem inteira.
