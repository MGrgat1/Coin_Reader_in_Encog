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

	class ImagePair {
		private final File file;
		private final int identity;

		public ImagePair(final File file, final int identity) {
			super();
			this.file = file;
			this.identity = identity;
		}

		public File getFile() {
			return this.file;
		}

		public int getIdentity() {
			return this.identity;
		}
	}


	private final List<ImagePair> imageList = new ArrayList<ImagePair>();

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
	 * Maps the identity string into the expected output neuron.
	 * For example: the identity "dollar" needs to be mapped to an output neuron.
	 * @param identity
	 * @return
	 */
	private int assignIdentity(final String identity) {

		if (this.identityToNeuronMap.containsKey(identity.toLowerCase())) {
			return this.identityToNeuronMap.get(identity.toLowerCase());
		}

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
	 * This is executed when the command is "input"
	 * Example: "Input: image:./coins/dollar.png, identity:dollar"
	 * @throws IOException
	 */
	public void processInput() throws IOException {
		final String image = getArg("image");
		final String identity = getArg("identity");

		final int idx = assignIdentity(identity);
		final File file = new File(image);

		this.imageList.add(new ImagePair(file, idx));

		System.out.println("Added input image:" + image);
	}

	public void processNetwork() throws IOException {
		System.out.println("Downsampling images...");

		for (final ImagePair pair : this.imageList) {
			final MLData ideal = new BasicMLData(this.outputCount);
			final int idx = pair.getIdentity();
			for (int i = 0; i < this.outputCount; i++) {
				if (i == idx) {
					ideal.setData(i, 1);
				} else {
					ideal.setData(i, -1);
				}
			}

			final Image img = ImageIO.read(pair.getFile());
			final ImageMLData data = new ImageMLData(img);
			this.training.add(data, ideal);
		}

		final String strHidden1 = getArg("hidden1");
		final String strHidden2 = getArg("hidden2");

		this.training.downsample(this.downsampleHeight, this.downsampleWidth);

		final int hidden1 = Integer.parseInt(strHidden1);
		final int hidden2 = Integer.parseInt(strHidden2);

		this.network = EncogUtility.simpleFeedForward(this.training
				.getInputSize(), hidden1, hidden2,
				this.training.getIdealSize(), true);
		System.out.println("Created network: " + this.network.toString());
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
