from dump import dump_main
from load import load_main
from execute import execute_main


# dumper
rxfn = 'tworay_rx_pts.txt'
txfn = 'tworay_tx_pts.txt'
envfn = 'env_firstmodel.txt'
dump_main(rxfn,txfn,envfn)

# raytracer
execute_main()

# loader
#load_main(['tworayresult'], testper = 0.1, verbose = 0, traindataname = 'tworaytrainset5in', testdataname = 'tworaytestset5in',extend=1)
load_main(['tworayresult'], testper = 0.5, verbose = 0, traindataname = 'tworaytrainseth301', testdataname = 'tworaytestseth301',extend=0)
