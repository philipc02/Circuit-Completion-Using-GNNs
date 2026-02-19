* SPICE netlist for the given circuit

I1 1 6 DC  /* Current source I_E connected between nodes 1 and 6 */
Q1 6 2 5 QMODEL  /* BJT with collector at 6, base at 2, and emitter at 5 */
RC 2 3 2k  /* Resistor R_C between nodes 2 and 3 */
V1 3 4 DC 5  /* Voltage source 5V between nodes 4 and 3 */

* Additional BJT model declaration, for simulation purposes
.model QMODEL NPN (BF=100)  /* Example model parameters */