plaintext
* List of components
* Q1, Q2, Q3, Q4 - NPN BJTs
* V1 - Voltage source
* I1, I2, I3 - Current sources
* R1, R2, R3 - Resistors

* Circuit definition
Q1 3 8 6 QNPN
Q2 6 5 0 QNPN
Q3 4 5 0 QNPN
Q4 2 9 7 QNPN

V1 8 0 DC 6

I1 6 0 DC 10uA
I2 4 0 DC 1mA
I3 2 0 DC 10uA

R1 3 12 10k
R2 3 11 5k
R3 7 11 5k
R4 9 0 10k

* Models
.model QNPN NPN(IS=1E-14 BF=100)

* Simulation commands
.TRAN 1u 100u
.END