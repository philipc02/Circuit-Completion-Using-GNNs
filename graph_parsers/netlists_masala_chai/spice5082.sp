plaintext
* Voltage Follower Circuit

Vin in 0 DC 1V
X1 in out out opamp

.subckt opamp 1 2 3
* 1 = non-inverting input
* 2 = inverting input
* 3 = output
e1 3 0 1 2 100k
r1 3 2 100MEG
c1 3 2 10p
.ends opamp

Vout out 0
.tran 1u 100u
.end