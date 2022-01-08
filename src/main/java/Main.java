import org.encog.Encog;

public class Main {

    /**
     * Creates an image neural network and executes a script contained within an input file
     * @param args Path to the input file (example: src/main/coins/input_file.txt)
     */
    public static void main(final String[] args) {
        if (args.length < 1) {
            System.out
                    .println("Must specify command file.  See source for format.");
        } else {
            System.out.println("[INFO] Accepted arguments");
            try {
                final ScriptProcessor scriptProcessor = new ScriptProcessor();
                System.out.println("[INFO] Created script execution module");
                System.out.println("[INFO] Entering program execution");
                scriptProcessor.execute(args[0]);
            } catch (final Exception e) {
                e.printStackTrace();
            }
        }

        Encog.getInstance().shutdown();
    }
}
