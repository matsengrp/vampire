import sys
import time

import click
import delegator
import pexpect


@click.command()
@click.option('--clusters', default='beagle', help="Which cluster to submit to.")
@click.argument('command')
def cli(clusters, command):
    """
    This includes doing filters as well as deduplicating on vjcdr3s.
    """

    max_n_retries = 10
    timeout = 60

    retry = 0
    command += f' --clusters={clusters}'
    click.echo(f"Running `{command}`.")

    main_command = None

    while True:

        if not main_command:
            main_command = pexpect.spawn(command, logfile=sys.stdout.buffer)

        try:
            main_command.expect("scons: done building targets", timeout=timeout)
            click.echo("SCons has completed.")
            break
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            click.echo("Process stopped without SCons completing.")

        c = delegator.run(f'squeue -u matsen -M {clusters} | tail -n +3')
        jobs_remaining = c.out.rstrip().split('\n')
        if len(jobs_remaining) > 0:
            print('\n'.join(jobs_remaining))
            click.echo("Re-expecting.")
            continue

        if main_command.exitstatus != 0:
            click.echo("Exiting with an error.")
            sys.exit(main_command.exitstatus)

        if retry < max_n_retries:
            retry = retry + 1
            click.echo(f"Things seem stuck. Restarting for retry {retry} of {max_n_retries}...")
            if not main_command.terminate():
                main_command.terminate(force=True)
            main_command = None


if __name__ == '__main__':
    cli()
