import static org.junit.Assert.*;

import org.junit.Test;

import struct.types.SLFeatureVector;

public class FeatureVectorTest {

	@Test
	public void test() {
		//fail("Not yet implemented");
	}
	
	@Test
	public void testDistFeatureVector() {
		SLFeatureVector fv1 = new SLFeatureVector(-1, -1.0, null);
		fv1.add(0, 1);
		fv1.add(1, 1);
		fv1.add(2, 1);
		SLFeatureVector fv2 = new SLFeatureVector(-1, -1.0, null);
		fv2.add(0, 4);
		fv2.add(5, 2);
		SLFeatureVector dv1 = SLFeatureVector.getDistVector(fv1, fv1);
		System.out.println(SLFeatureVector.size(dv1));
		System.out.println(dv1.toString());
		
	}

}
