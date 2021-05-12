
Directory Setup
```
scratch/$USER (max 50TB, for temp data, 30 day limit)
home/$USER (max 20GB)
data/$USER (max 250GB)
```

View exact quota
```
pgquota
```

Submit job
```
sbatch job.sh
```

View job status
```
squeue -u $USER
```

Send file from local to peregrine
```
scp -r [directory] [username]@peregrine.hpc.rug.nl:/scratch/[username]/[directory]
```