* Components
V1 7 0 DC 0
RS 5 2 1k
C1 2 4 0.1uF
RE 4 0 10k
V2 4 0 DC 20
CC1 2 3 0.1uF
RC 3 6 6.5k
V3 6 0 DC -25
CC2 3 8 0.1uF
RL 8 0 5k

* Voltage source
VI 7 5 DC

* Analysis
.TRAN 1m 100m
.END