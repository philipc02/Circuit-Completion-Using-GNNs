plaintext
* BJT Amplifier Circuit
V1 7 0 DC 0V
Vi 7 0 DC
VS 5 0 DC 5V
VEE 3 0 DC -5V

RS 7 2 0.1k
R1 5 4 40k
R2 2 3 5.7k
RC 4 6 5k
RE 3 0 0.5k
RL 6 0 10k

CC 2 7 10u
CL 6 0 15p

Q1 4 2 3 NPN

.MODEL NPN NPN (IS=1E-14 BF=100)

.END