spice
* Circuit Description
V1 1 0 AC 0
C1 1 3 1uF
D1 3 4 D
V2 4 0 DC 5
R1 3 5 1k
.model D D
.tran 0.1us 10ms
.end