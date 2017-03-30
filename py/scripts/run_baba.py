import sys
from baba import decompose_z_baba_abba

def make_z_baba_abba(in_file, sorted_pops_file, n_components, l1_penalty,
                     outdir, seed=None):
    inferred_components_file = os.path.join(outdir, "inferred_components.txt")
    decompose_z_baba_abba(in_file, n_components, l1_penalty,
                          os.path.join(outdir, "optimization_result.json"),
                          inferred_components_file)
    subprocess.check_call(["Rscript", "R/scripts/run_baba.R", "plot_baba_matrices",
                           inferred_components_file, sorted_pops_file,
                           os.path.join(outdir, "z_baba_abba_squares."),
                           "Pop1", "Pop3", "Pop2", "Pop4"])
    subprocess.check_call(["Rscript", "R/scripts/run_baba.R", "plot_baba_vectors",
                           inferred_components_file, sorted_pops_file,
                           os.path.join(outdir, "z_baba_abba_vectors.pdf")])

def clean_qpDstats_output():
    for line in sys.stdin:
        if line.startswith("result:"):
            print(*line.split()[1:], sep="\t")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
