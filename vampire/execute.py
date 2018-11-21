"""
Execute commands locally or via sbatch.
"""

import click
import os
import subprocess
import time
import uuid

sbatch_prelude = """#!/bin/bash
#SBATCH -c 18
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p largenode
#SBATCH --mem=30000
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsen@fredhutch.org
set -eu
set -o pipefail
hostname
source activate py36
cd /home/matsen/re/vampire/vampire/
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
@click.argument('sources')
@click.argument('targets')
@click.argument('to_execute_f_string')
def cli(clusters, sources, targets, to_execute_f_string):
    """
    Execute a command with certain sources and targets, perhaps on a SLURM
    cluster via sbatch. Wait until the command has completed.

    SOURCES: Input files as a space-separated list.

    TARGETS: Output files as a space-separated list.

    TO_EXECUTE_F_STRING: The command to execute, where '{sources}' gets
    replaced by the sources argument, and '{targets}' gets replaced by the
    targets argument.
    """

    if clusters == '':
        # Local execution.
        to_execute = to_execute_f_string.format(sources=sources, targets=targets)
        click.echo("Executing locally:")
        click.echo(to_execute)
        return subprocess.check_output(to_execute, shell=True)

    job_uuid = uuid.uuid4().hex

    if clusters == 'beagle':
        # Put the data where beagle likes it.
        beagle_input_dir = os.path.join('/mnt/beagle/delete10/matsen_e/vampire/uuid', job_uuid)
        sources_l, cp_instructions = translate_paths(sources.split(), beagle_input_dir)
        cp_instructions = [f'mkdir -p {beagle_input_dir}'] + list(cp_instructions)
        sources = ' '.join(sources_l)
    else:
        cp_instructions = []

    # Put the batch script in the directory of the first target.
    execution_dir = os.path.dirname(targets.split()[0])
    script_name = 'job.sh'
    sentinel_path = os.path.join(execution_dir, 'sentinel.' + job_uuid)
    with open(os.path.join(execution_dir, script_name), 'w') as fp:
        fp.write(sbatch_prelude)
        for instruction in cp_instructions:
            fp.write(instruction + '\n')
        fp.write(to_execute_f_string.format(sources=sources, targets=targets) + '\n')
        fp.write(f'touch {sentinel_path}\n')

    out = subprocess.check_output(f'cd {execution_dir}; sbatch --clusters {clusters} {script_name}', shell=True)
    click.echo(out.decode('UTF-8'))

    # Wait until the sentinel file appears, then clean up.
    while not os.path.exists(sentinel_path):
        time.sleep(5)
    os.remove(sentinel_path)

    return out


if __name__ == '__main__':
    cli()
