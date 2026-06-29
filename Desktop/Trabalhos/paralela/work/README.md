## Como compilar

Requer MPI instalado e carregado na máquina. Após isso, basta executar o `make`.

```bash
$ make
```

## Como executar

É necessário redirecionar a saída padrão (`stdout`) para um arquivo no formato PPM.
Outras informações printadas pelo código são direcionadas à saída de erro (`stderr`).

```bash
$ mpirun -n <n> ./build/main <width> > image.ppm
```
