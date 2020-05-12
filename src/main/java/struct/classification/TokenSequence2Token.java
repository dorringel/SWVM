package struct.classification;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.TokenSequence;

/**
 * Convert the length-one TokenSequence to a Token. Exception occurs if the
 * TokenSequence is length more than one.
 *
 * @author Kuzman Ganchev
 */

public class TokenSequence2Token extends Pipe {

	public TokenSequence2Token() {
	}

	@Override
	public Instance pipe(Instance carrier) {
		TokenSequence ts = (TokenSequence) carrier.getData();
		if (ts.size() != 1) {
			throw new IllegalArgumentException("Instance has " + ts.size() + " tokens");
		}
		carrier.setData(ts.get(0));
		LabelSequence ls = (LabelSequence) carrier.getTarget();
		carrier.setTarget(ls.getLabelAtPosition(0));
		return carrier;
	}

	// Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject(ObjectOutputStream out) throws IOException {
		out.writeInt(CURRENT_SERIAL_VERSION);
	}

	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt();
	}
}
