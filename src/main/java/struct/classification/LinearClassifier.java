/**
 *
 */
package struct.classification;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelVector;
import struct.alg.Predictor;
import struct.types.SLFeatureVector;

/**
 * The Linear Classifier.
 *
 * @version 08/20/2006
 */
public class LinearClassifier extends Classifier implements Serializable {

	private static final long serialVersionUID = 4395890591236322275L;

	private Predictor predictor;
	private LabelAlphabet tagAlphabet;
	private Alphabet phiAlphabet;

	public LinearClassifier() {
	}

	public LinearClassifier(Pipe dataPipe) {
		super(dataPipe);
		tagAlphabet = (LabelAlphabet) dataPipe.getTargetAlphabet();
		phiAlphabet = new Alphabet();
	}

	/**
	 * Classifies the instance according to this classifier.
	 */
	@Override
	public Classification classify(Instance instance) {
		LabelAlphabet la = getLabelAlphabet();
		int numClasses = la.size();
		double[] scores = new double[numClasses];
		getClassificationScores(instance, scores);

		return new Classification(instance, this, new LabelVector(la, scores));
	}

	private void getClassificationScores(Instance instance, double[] scores) {
		ClassificationInstance inst = getInstance(instance);
		ClassificationPrediction prediction = (ClassificationPrediction) predictor.decode(inst, inst.getFeatures());

		LabelAlphabet la = getLabelAlphabet();

		double predictedScores[] = prediction.getScores();

		for (int i = 0; i < la.size(); i++) {
			String tag = prediction.getLabelByRank(i).getTag();
			int index = la.lookupIndex(tag);
			scores[index] = predictedScores[i];
		}
	}

	/**
	 * Adds features to the phiAlphabet.
	 */
	public void createPhiAlphabet(InstanceList trainingSet) {
		Instance inst;
		FeatureVector fv;
		Label label;
		for (int i = 0; i < trainingSet.size(); i++) {
			inst = trainingSet.get(i);
			fv = (FeatureVector) inst.getData();
			label = (Label) inst.getTarget();

			String tag = (String) label.getEntry();

			createFeatureVector(fv, tag, new SLFeatureVector(-1, -1.0, null));

			createU(fv);
		}
		phiAlphabet.stopGrowth();

		if (predictor == null)
			predictor = new ClassificationPredictor(phiAlphabet.size());
	}

	private SLFeatureVector createFeatureVector(FeatureVector featureVector, String next, SLFeatureVector fv) {

		int[] indices = featureVector.getIndices();
		String s2 = next;
		for (int j = 0; j < indices.length; j++) {
			String pred = "feat" + indices[j];
			fv = fv.add(s2 + "_" + pred, 1.0, phiAlphabet);
		}
		return fv;
	}

	private void createU(FeatureVector featureVector) {
		for (int j = 0; j < tagAlphabet.size(); j++) {
			createFeatureVector(featureVector, (String) tagAlphabet.lookupObject(j),
					new SLFeatureVector(-1, -1.0, null));
		}
	}

	public void setInstanceAlphabets() {
		ClassificationInstance.setTagAlphabet(tagAlphabet);
		ClassificationInstance.setDataAlphabet(phiAlphabet);
	}

	private ClassificationInstance getInstance(Instance inst) {
		FeatureVector fv = (FeatureVector) inst.getData();

		String tag = ((Label) inst.getTarget()).toString();

		return new ClassificationInstance(tag, fv);
	}

	/**
	 * Expands the classifier for training with new features.
	 *
	 * @param tagAlphabet
	 *            - the expanded labelAlphabet
	 * @param trainingSet
	 *            - the new training set
	 */
	public void grow(Alphabet tagAlphabet, InstanceList trainingSet) {
		if (this.tagAlphabet != tagAlphabet)
			throw new IllegalArgumentException("Cannot use a different tagAlphabet!");

		growPhiAlphabet(trainingSet);
	}

	private void growPhiAlphabet(InstanceList trainingSet) {
		phiAlphabet.startGrowth();

		Instance inst;
		FeatureVector fv;
		Label label;
		for (int i = 0; i < trainingSet.size(); i++) {
			inst = trainingSet.get(i);
			fv = (FeatureVector) inst.getData();
			label = (Label) inst.getTarget();

			String tag = (String) label.getEntry();

			createFeatureVector(fv, tag, new SLFeatureVector(-1, -1.0, null));

			createU(fv);
		}

		phiAlphabet.stopGrowth();

		predictor.grow(phiAlphabet.size());
	}

	public Predictor getPredictor() {
		return predictor;
	}

	private void writeObject(ObjectOutputStream stream) throws IOException {
		stream.writeObject(instancePipe);
		stream.writeObject(predictor);
		stream.writeObject(phiAlphabet);
		stream.writeObject(tagAlphabet);
	}

	private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
		instancePipe = (Pipe) stream.readObject();
		predictor = (Predictor) stream.readObject();
		phiAlphabet = (Alphabet) stream.readObject();
		tagAlphabet = (LabelAlphabet) stream.readObject();
		ClassificationInstance.setDataAlphabet(phiAlphabet);
		ClassificationInstance.setTagAlphabet(tagAlphabet);
	}

	public Pipe getPipe() {
		return instancePipe;
	}
}
