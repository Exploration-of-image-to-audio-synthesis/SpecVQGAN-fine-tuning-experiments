# Interactive work

First you reserve resources with `srun —pty bash`

This launches a single core interactive bash shell on the compute node. Here you can do interactive work

The command right there is ok but it might be terminated soon, so better use something like `srun --interactive -c1 -n 4 --mem=4G -t 00:10:00 -p short -M [ukko|kale|carrington] --pty bash`

where
* c1 - number of cpus (1)
* n - number of tasks (4)
* mem - memory
* t - time
* p - partition
* M - cluster