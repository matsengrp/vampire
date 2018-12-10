"""
Execute commands locally or via sbatch/singularity.

For sbatch/singularity, we write out two files. First, the "run" script that
actually runs the specified command, and second, the "batch" script that gets
submitted to the cluster.
"""

import click
import os
import subprocess
import time
import uuid
import re

batch_prelude = """#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p campus
#SBATCH -o {execution_dir}/job_cpu_%j.out
#SBATCH -e {execution_dir}/job_cpu_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsen@fredhutch.org
set -eu

source /app/Lmod/lmod/lmod/init/zsh
export LMOD_PACKAGE_PATH=/app/Lmod
module use /app/easybuild/modules/all
module use /app/Modules/modulefiles

module load Singularity/2.5.2-GCC-5.4.0-2.26
"""

run_prelude = """#!/bin/bash
set -eux

pip install -e /home/matsen/re/vampire
"""


def translate_paths(in_paths, dest_dir):
    """
    Copy all files in in_paths to dest_dir and return a tuple consisting of
    a list of their new paths and a list of cp instructions about how to get
    them there.
    """

    def aux():
        for path in in_paths:
            fname = os.path.basename(path)
            dest_path = os.path.join(dest_dir, fname)
            yield dest_path, f'cp {path} {dest_path}'

    return zip(*aux())


@click.command()
@click.option('--clusters', default='', help='Clusters to submit to. Default is local execution.')
@click.option('--script-prefix', default='job', help='Prefix for job script name.')
@click.argument('sources')
@click.argument('targets')
@click.argument('to_execute_f_string')
def cli(clusters, script_prefix, sources, targets, to_execute_f_string):
    """
    Execute a command with certain sources and targets, perhaps on a SLURM
    cluster via sbatch. Wait until the command has completed.

    SOURCES: Input files as a space-separated list.

    TARGETS: Output files as a space-separated list.

    TO_EXECUTE_F_STRING: The command to execute, where '{sources}' gets
    replaced by the sources argument, and '{targets}' gets replaced by the
    targets argument.
    """

    # Remove all quotes: they can get in the way with our basename noodling.
    sources = re.sub('"*\'*', '', sources)
    targets = re.sub('"*\'*', '', targets)

    if clusters == '':
        # Local execution.
        to_execute = to_execute_f_string.format(sources=sources, targets=targets)
        click.echo("Executing locally:")
        click.echo(to_execute)
        return subprocess.check_output(to_execute, shell=True)

    job_uuid = uuid.uuid4().hex
    cluster_directory_d = {
        'beagle': '/mnt/beagle/delete10/matsen_e/vampire/uuid',
        'koshu': '/fh/scratch/delete30/matsen_e/vampire/uuid'
    }

    if clusters in cluster_directory_d:
        # Put the data where the cluster likes it.
        input_dir = os.path.join(cluster_directory_d[clusters], job_uuid)
        sources_l, cp_instructions = translate_paths(sources.split(), input_dir)
        cp_instructions = [f'mkdir -p {input_dir}'] + list(cp_instructions)
        sources = ' '.join(sources_l)
    else:
        cp_instructions = []

    # Put the batch script in the directory of the first target.
    execution_dir = os.path.dirname(targets.split()[0])
    sentinel_path = os.path.join(execution_dir, 'sentinel.' + job_uuid)
    run_path = os.path.join(execution_dir, f'{script_prefix}.{job_uuid}.run.sh')
    with open(run_path, 'w') as fp:
        fp.write(run_prelude)
        for instruction in cp_instructions:
            fp.write(instruction + '\n')
        fp.write(to_execute_f_string.format(sources=sources, targets=targets) + '\n')
        fp.write(f'touch {sentinel_path}\n')
    # This chmod appears necessary for `singularity run` to work.
    os.chmod(run_path, 0o755)

    batch_path = os.path.join(execution_dir, f'{script_prefix}.{job_uuid}.batch.sh')
    with open(batch_path, 'w') as fp:
        fp.write(batch_prelude.format(execution_dir=execution_dir))
        fp.write("singularity exec docker://matsen/vampire " + run_path + '\n')

    out = subprocess.check_output(f'sbatch --clusters {clusters} {batch_path}', shell=True)
    click.echo(out.decode('UTF-8'))

    # Wait until the sentinel file appears, then clean up.
    while not os.path.exists(sentinel_path):
        time.sleep(5)
    os.remove(sentinel_path)

    return out


if __name__ == '__main__':
    cli()
