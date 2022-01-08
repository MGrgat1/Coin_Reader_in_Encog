/*
 * Encog(tm) Examples v3.0 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */

import org.encog.EncogError;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.platformspecific.j2se.TrainingDialog;
import org.encog.platformspecific.j2se.data.image.ImageMLData;
import org.encog.platformspecific.j2se.data.image.ImageMLDataSet;
import org.encog.util.downsample.Downsample;
import org.encog.util.downsample.RGBDownsample;
import org.encog.util.downsample.SimpleIntensityDownsample;
import org.encog.util.simple.EncogUtility;

import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;



public class ImageNeuralNetwork {

	private final Map<String, String> args = new HashMap<String, String>();
	private String line;

	class TrainingPair {
		private final File file;
		private final int outputNeuronValue;

		public TrainingPair(final File file, final int outputNeuronValue) {
			super();
			this.file = file;
			this.outputNeuronValue = outputNeuronValue;
		}

		public File getFile() {
			return this.file;
		}

		public int getOutputNeuronValue() {
			return this.outputNeuronValue;
		}

		@Override
		public String toString() {
			return "TrainingPair{" +
					"file=" + file +
					", outputNeuronValue=" + outputNeuronValue +
					'}';
		}
	}


	private final List<TrainingPair> imageList = new ArrayList<TrainingPair>();

	private final Map<String, Integer> identityToNeuronMap = new HashMap<String, Integer>();
	private final Map<Integer, String> neuronToIdentityMap = new HashMap<Integer, String>();
	private ImageMLDataSet training;

	private int outputCount;
	private int downsampleWidth;
	private int downsampleHeight;
	private BasicNetwork network;

	private Downsample downsample;

	public String getArg(final String name) {
		final String result = this.args.get(name);
		if (result == null) {
			throw new EncogError("Missing argument " + name + " on line: "
					+ this.line);
		}
		return result;
	}

	/**
	 * Associates each input (dollar, dime, cent, etc.) with a possible output neuron
	 * For example: the identity "dollar" needs to be mapped to an output neuron.
	 * @param identity the objects that the network needs to recognize ('dollar', 'dime', 'cent')
	 * @return the integer value that output neurons will have when they recognize that object
	 */
	private int assignOutputNeuron(final String identity) {

		//if we had already assigned an output neuron to this value, give it that value again
		if (this.identityToNeuronMap.containsKey(identity.toLowerCase())) {
			return this.identityToNeuronMap.get(identity.toLowerCase());
		}

		/*
		 * If this is the first time we have a dollar as an input, assign it a new output neuron.
		 * For example, if we have inputs 'dollar', 'dime', 'cent', their output neurons will be '0', '1', '2'.
		 * And from then on, whenever the network outputs a '2', it will mean that it recognized a cent
		 * (either correctly or incorrectly).
		 */

		final int result = this.outputCount;
		this.identityToNeuronMap.put(identity.toLowerCase(), result);
		this.neuronToIdentityMap.put(result, identity.toLowerCase());
		this.outputCount++;
		return result;
	}


	public void processCreateTraining() {
		final String strWidth = getArg("width");
		final String strHeight = getArg("height");
		final String strType = getArg("type");

		this.downsampleHeight = Integer.parseInt(strHeight);
		this.downsampleWidth = Integer.parseInt(strWidth);

		if (strType.equals("RGB")) {
			this.downsample = new RGBDownsample();
		} else {
			this.downsample = new SimpleIntensityDownsample();
		}

		this.training = new ImageMLDataSet(this.downsample, false, 1, -1);
		System.out.println("Training set created");
	}

	/**
	 * Assign an output value for every possible input file
	 * For example: ("./coins/dollar.png", 0) - 0 means that the network has recognized a dollar
	 * @throws IOException
	 */
	public void processInput() throws IOException {
		final String image = getArg("image");
		final String identity = getArg("identity");

		final int outputNeuronValue = assignOutputNeuron(identity);
		final File file = new File(image);

		TrainingPair trainingPair = new TrainingPair(file, outputNeuronValue);

		this.imageList.add(trainingPair);

		System.out.println("[INFO] Added a new training pair: " + trainingPair);
	}

	/**
	 * Creates the expected output, reads the image from the image file, creates the hidden layers,
	 * downsamples the images, and performs the feedforward algorithm.
	 * @throws IOException
	 */
	public void processCreateNetwork() throws IOException {

		System.out.println("[INFO] Creating expected output");
		for (final TrainingPair pair : this.imageList) {
			final MLData expectedOutput = new BasicMLData(this.outputCount);
			final int outputNeuronValue = pair.getOutputNeuronValue();

			/**
			 * All outputs of the neural network will have 1 in one place, and -1 everywhere else. The network's guess
			 * will be encoded by the position of that value.
			 */
			for (int i = 0; i < this.outputCount; i++) {
				if (i == outputNeuronValue) {
					expectedOutput.setData(i, 1);
				} else {
					expectedOutput.setData(i, -1);
				}
			}

			System.out.println("[INFO] Reading the image from the image file");
			final Image img = ImageIO.read(pair.getFile());
			final ImageMLData data = new ImageMLData(img);
			this.training.add(data, expectedOutput);
		}

		final String strHidden1 = getArg("hidden1");
		final String strHidden2 = getArg("hidden2");

		System.out.println("[INFO] Downsampling images...");

		this.training.downsample(this.downsampleHeight, this.downsampleWidth);

		final int hidden1 = Integer.parseInt(strHidden1);
		final int hidden2 = Integer.parseInt(strHidden2);

		this.network = EncogUtility.simpleFeedForward(this.training
				.getInputSize(), hidden1, hidden2,
				this.training.getIdealSize(), true);
		System.out.println("[INFO] Created network: " + this.network.toString());
	}

	public void processTrain() throws IOException {
		final String strMode = getArg("mode");
		final String strMinutes = getArg("minutes");
		final String strStrategyError = getArg("strategyerror");
		final String strStrategyCycles = getArg("strategycycles");

		System.out.println("Training Beginning... Output patterns="
				+ this.outputCount);

		final double strategyError = Double.parseDouble(strStrategyError);
		final int strategyCycles = Integer.parseInt(strStrategyCycles);

		final ResilientPropagation train = new ResilientPropagation(this.network, this.training);
		train.addStrategy(new ResetStrategy(strategyError, strategyCycles));

		if (strMode.equalsIgnoreCase("gui")) {
			TrainingDialog.trainDialog(train, this.network, this.training);
		} else {
			final int minutes = Integer.parseInt(strMinutes);
			EncogUtility.trainConsole(train, this.network, this.training,
					minutes);
		}
		System.out.println("Training Stopped...");
	}

	public void processWhatIs() throws IOException {
		final String filename = getArg("image");
		final File file = new File(filename);
		final Image img = ImageIO.read(file);
		final ImageMLData input = new ImageMLData(img);
		input.downsample(this.downsample, false, this.downsampleHeight,
				this.downsampleWidth, 1, -1);
		final int winner = this.network.winner(input);
		System.out.println("What is: " + filename + ", it seems to be: "
				+ this.neuronToIdentityMap.get(winner));
	}
}
