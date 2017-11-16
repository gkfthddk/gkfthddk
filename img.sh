#!/bin/sh

for number in {1..50}
do
RUN="pt_100_500_$number"
#GINAME="raw/mg5_pp_gg_default_$RUN.root"
GONAME="mg5_pp_zg_passed_$RUN"
#QINAME="raw/mg5_pp_qq_default_$RUN.root"
QONAME="mg5_pp_zq_passed_$RUN"

./append $GONAME
./append $QONAME

done
