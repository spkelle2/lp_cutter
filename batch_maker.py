
def make_pbs_files(ns, base_name):
    for n in ns:
        txt = f"""#PBS -N {base_name}_{n}
#PBS -e /home/sek519/lp_cutter/{base_name}_{n}.err
#PBS -o /home/sek519/lp_cutter/{base_name}_{n}.out
#PBS -l nodes=1:ppn=4,mem={4 if n <= 100 else 12}gb,vmem={6 if n <= 100 else 18}gb
#PBS -q {'verylong' if n > 150 else 'long' if n > 50 else 'medium'}

cd /home/sek519/lp_cutter
source /home/sek519/lp_cutter/env_lpc/bin/activate
python3 runner.py {base_name}_{n} {n}"""
        text_file = open(f"{base_name}_{n}.pbs", "w")
        text_file.write(txt)
        text_file.close()


if __name__ == '__main__':
    make_pbs_files(range(10, 110, 10), 'small_random_pp')
