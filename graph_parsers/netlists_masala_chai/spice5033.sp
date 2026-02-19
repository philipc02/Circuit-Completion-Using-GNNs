plaintext
* SPICE Netlist for BJT Amplifier Circuit

VCC 4 0 DC <VCC_VALUE>
VEE 6 0 DC <VEE_VALUE>
V1 5 0 AC <V1_AC_VALUE> DC <V1_DC_OFFSET>

RC 4 Vout <RC_VALUE>
RE 6 3 <RE_VALUE>

Q1 2 5 6 NPN_MODEL
Q2 3 2 4 NPN_MODEL

.model NPN_MODEL NPN

.control
run
.endc
.end