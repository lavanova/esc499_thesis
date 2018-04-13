from dump import dump_main
from load import load_main
from execute import execute_main


# dumper
rxfn = 'urbancanyon_rx_pts.txt'
txfn = 'urbancanyon_tx_pts.txt'
envfn = 'env_urbancanyon.txt'
dump_main(rxfn,txfn,envfn)

# raytracer
execute_main()

# loader
load_main(['urbancanyonresult'], testper = 0.5, verbose = 0, traindataname = 'uctrainsetr3', testdataname = 'uctestsetr3')
