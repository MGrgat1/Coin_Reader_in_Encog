import org.encog.EncogError;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * This is how an input file should be structured:
 *
 * CreateTraining: width:16,height:16,type:RGB
 * Input: image:./coins/dime.png, identity:dime
 * Input: image:./coins/dollar.png, identity:dollar
 * Input: image:./coins/half.png, identity:half dollar
 * Input: image:./coins/nickle.png, identity:nickle
 * Input: image:./coins/penny.png, identity:penny
 * Input: image:./coins/quarter.png, identity:quarter
 * Network: hidden1:100, hidden2:0
 * Train: Mode:console, Minutes:1, StrategyError:0.25, StrategyCycles:50
 * Whatis: image:./coins/dime.png
 * Whatis: image:./coins/half.png
 * Whatis: image:./coins/testcoin.png
 *
 */

public class ScriptProcessor {

    private final Map<String, String> args = new HashMap<String, String>();
    private String line;

    ImageNeuralNetwork imageNeuralNetwork = new ImageNeuralNetwork();

    public ScriptProcessor() {
    }

    /**
     * Executes the commands stored in an input file (Example file: src/main/coins/input_file.txt)
     * @param file
     * @throws IOException
     */
    public void execute(final String file) throws IOException {
        System.out.println("[INFO] Entered program execution");
        final FileInputStream fstream = new FileInputStream(file);
        final DataInputStream in = new DataInputStream(fstream);
        final BufferedReader br = new BufferedReader(new InputStreamReader(in));

        System.out.println("[INFO] Entering line readings");
        while ((this.line = br.readLine()) != null) {
            System.out.println("[INFO] Successfully read line:");
            System.out.println(this.line);
            System.out.println("[INFO] Entering line execution");
            executeLine();
        }
        in.close();
    }

    /**
     * Execute a command based on the name of the command, and the arguments given to it in the input file.
     * Example:
     * command - "Input",
     * args - ("image", "./coins/dime.png"), ("identity", "dime")
     *
     * @throws IOException
     */
    public void executeCommand(final String command,
                                final Map<String, String> args) throws IOException {
        if (command.equals("input")) {
            imageNeuralNetwork.processInput();
        } else if (command.equals("createtraining")) {
            imageNeuralNetwork.processCreateTraining();
        } else if (command.equals("train")) {
            imageNeuralNetwork.processTrain();
        } else if (command.equals("network")) {
            imageNeuralNetwork.processNetwork();
        } else if (command.equals("whatis")) {
            imageNeuralNetwork.processWhatIs();
        }

    }

    /**
     * Executes a line read from the input file.
     *
     * Example lines:
     * CreateTraining: width:16,height:16,type:RGB
     * Input: image:./coins/dime.png, identity:dime
     * Input: image:./coins/dollar.png, identity:dollar
     * Input: image:./coins/half.png, identity:half dollar
     * Input: image:./coins/nickle.png, identity:nickle
     * Input: image:./coins/penny.png, identity:penny
     * Input: image:./coins/quarter.png, identity:quarter
     * Network: hidden1:100, hidden2:0
     * Train: Mode:console, Minutes:1, StrategyError:0.25, StrategyCycles:50
     * Whatis: image:./coins/dime.png
     * Whatis: image:./coins/half.png
     * Whatis: image:./coins/testcoin.png
     *
     * These lines are all read and executed one after another.
     *
     * @throws IOException
     */
    public void executeLine() throws IOException {
        //if the line doesn't contain a phrase with a ':' char (a phrase like "Input:"), the line is not a proper command
        final int index = this.line.indexOf(':');
        if (index == -1) {
            throw new EncogError("Invalid command: " + this.line);
        }

        // Separate the line into a command and its arguments
        // Example: "Input: image:./coins/penny.png, identity:penny" is separated into
        // command - "Input"
        // argsStr - "image:./coins/penny.png, identity:penny"
        final String command = this.line.substring(0, index).toLowerCase()
                .trim();
        final String argsStr = this.line.substring(index + 1).trim();

        // Split the arguments into tokens that represent the arguments
        // Example: "image:./coins/dime.png, identity:dime" is tokenized into ("image:./coins/dime.png", "identity:dime")
        final StringTokenizer tok = new StringTokenizer(argsStr, ",");
        this.args.clear();
        while (tok.hasMoreTokens()) {

            // Example: "image:./coins/dime.png"
            final String arg = tok.nextToken();

            // if the argument doesn't contain the ':' char, then it's not a valid argument
            final int index2 = arg.indexOf(':');
            if (index2 == -1) {
                throw new EncogError("Invalid command: " + this.line);
            }

            // Separate the argument into a key and its value
            // Example: "image:./coins/penny.png" is separated into
            // key - "image"
            // value - "./coins/penny.png"
            final String key = arg.substring(0, index2).toLowerCase().trim();
            final String value = arg.substring(index2 + 1).trim();

            //all the key-value pairs from the command will be placed into args and used to execute the command
            this.args.put(key, value);
        }

        // the command will be executed based on the name of the command and the arguments that followed its name
        // Example:
        // command - "Input",
        // args - ("image", "./coins/dime.png"), ("identity", "dime")
        executeCommand(command, this.args);
    }


}
