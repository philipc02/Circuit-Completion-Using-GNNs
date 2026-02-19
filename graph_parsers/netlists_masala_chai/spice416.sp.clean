spice
* Operational Amplifier
.subckt opamp_plus_minus 2 5 4
* inverting input: 2, non-inverting input: 5, output: 4
Rin 2 in 1MEG
Eout 4 0 in 5 100k
.ends

* Resistors
R1 3 0 R1_value
R2 4 6 R2_value

* Current Source
I1 3 0 I_t_value

* Connections
X1 2 5 4 opamp_plus_minus

* Net Definitions
* 0: Ground
* 2: Input to opamp (- terminal)
* 4: Output of opamp
* 5: Input to opamp (+ terminal)
* 6: Connecting node