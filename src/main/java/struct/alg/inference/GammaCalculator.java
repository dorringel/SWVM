/**
 *
 */
package struct.alg.inference;

import struct.types.SLLabel;

public interface GammaCalculator {

	/**
	 * Calculate the gamma function of a mixed assignment.
	 * 
	 * @param mJ
	 *            The target assignment to calculate the gamma of
	 * @return /gamma(mJ)
	 */
	public double calculate(SLLabel mJ);
}
