/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.solver;

import struct.types.SLFeatureVector;

/** QPConstraints.
 * 
 * @version 07/15/2006
 */
public class QPConstraints {
	
	public SLFeatureVector[] a;
	public SLFeatureVector[] a1;
	public SLFeatureVector[] a2;
	public double[] b;
	public int dim;
	public int dim1;
	public int dim2;
	public double C;
	
	public double[] w1;
	public double[] w2;
	public double[] alpha;
	
	public QPConstraints() {
		a1=null; a2=null; a=null;
		b=null;
		dim1=-1; dim2=-1; dim=-1;
		C = 1;
		w1=null;w2=null;alpha=null;
	}
	
	public String toString() {
		StringBuffer str = new StringBuffer();
		if (a1 != null)
			str.append("a1 length : " + a1.length  + " \n");
		else
			str.append("a1 is null\n");
		
		if (a2 != null)
			str.append("a2 length : " + a2.length + " \n");
		else
			str.append("a2 is null\n");
		
		if (a != null)
			str.append("a length : " + a.length + " \n");
		else
			str.append("a is null\n");
		
		if (b != null)
			str.append("b length : " + b.length + " \n");
		else
			str.append("b is null\n");
		
		if (w1 != null)
			str.append("w1 length : " + w1.length + " \n");
		else
			str.append("w1 is null\n");
		
		if (w2 != null)
			str.append("w2 length : " + w2.length + " \n");
		else
			str.append("w2 is null\n");
		
		if (alpha != null)
			str.append("alpha length : " + alpha.length + " \n");
		else
			str.append("alpha is null\n");
		
		str.append("dim : " + String.valueOf(dim) + "\n");
		str.append("dim1 : " + String.valueOf(dim1) + "\n");
		str.append("dim2 : " + String.valueOf(dim2) + "\n");
		str.append("C : " + String.valueOf(C) + "\n");
		return (str.toString());		
	}
};
