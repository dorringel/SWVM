package struct.tests;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.iterator.RandomTokenSequenceIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Dirichlet;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;
import junit.framework.TestCase;
import struct.classification.KBestMiraClassifierTrainer;

public class TestClassification extends TestCase {

	public static void main(String[] args) {
		junit.textui.TestRunner.run(TestClassification.class);
	}

	public TestClassification(String name) {
		super(name);
	}

	@Override
	protected void setUp() throws Exception {
		super.setUp();
	}

	@Override
	protected void tearDown() throws Exception {
		super.tearDown();
	}

	private static Alphabet dictOfSize(int size) {
		Alphabet ret = new Alphabet();
		for (int i = 0; i < size; i++)
			ret.lookupIndex("feature" + i);
		return ret;
	}

	public void testRandomTrained() {
		Alphabet fd = dictOfSize(3);
		String[] classNames = new String[] { "class0", "class1", "class2" };

		InstanceList ilist = new InstanceList(new Randoms(1), fd, classNames, 200);
		InstanceList lists[] = ilist.split(new java.util.Random(2), new double[] { .5, .5 });

		ClassifierTrainer trainer = new KBestMiraClassifierTrainer(1);

		Classifier classifier = trainer.train(lists[0]);

		System.out.println("Accuracy on training set:");
		System.out.println(classifier.getClass().getName() + ": " + new Trial(classifier, lists[0]).getAccuracy());

		System.out.println("Accuracy on testing set:");
		System.out.println(classifier.getClass().getName() + ": " + new Trial(classifier, lists[1]).getAccuracy());
	}

	public void testNewFeatures() {
		ClassifierTrainer trainer = new KBestMiraClassifierTrainer(1);

		Alphabet fd = dictOfSize(3);
		String[] classNames = new String[] { "class0", "class1", "class2" };

		Randoms r = new Randoms(1);
		InstanceList training = new InstanceList(r, fd, classNames, 50);
		expandDict(fd, 25);

		Classifier classifier = trainer.train(training);

		System.out.println("Accuracy on training set:");
		System.out.println(classifier.getClass().getName() + ": " + new Trial(classifier, training).getAccuracy());

		InstanceList testing = new InstanceList(training.getPipe());
		RandomTokenSequenceIterator iter = new RandomTokenSequenceIterator(r, new Dirichlet(fd, 2.0), 30, 0, 10, 50,
				classNames);
		testing.addThruPipe(iter);

		// for (int i = 0; i < testing.size (); i++) {
		// Instance inst = testing.getInstance (i);
		// System.out.println ("DATA:"+inst.getData());
		// }

		System.out.println("Accuracy on testing set:");
		System.out.println(classifier.getClass().getName() + ": " + new Trial(classifier, testing).getAccuracy());
	}

	private void expandDict(Alphabet fd, int size) {
		fd.startGrowth();
		for (int i = 0; i < size; i++)
			fd.lookupIndex("feature" + i, true);
	}
}
