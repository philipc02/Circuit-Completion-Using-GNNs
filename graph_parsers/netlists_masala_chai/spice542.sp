plaintext
* R and C values should be specified by the user
* Voltage source value should be specified by the user

V1 1 0 Vsd

R1 1 2 R
R2 0 2 R

C1 3 2 C
C2 2 0 C

* Assuming the op-amp is ideal, it doesn't have an explicit SPICE model
* Op-amp inputs are at nodes 2 and 0, output is at node 4

* Feedback configuration
XOPAMP 2 0 4 OPAMP

.END