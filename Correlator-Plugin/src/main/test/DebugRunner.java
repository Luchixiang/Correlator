import ij.ImageJ;

public class DebugRunner {

    public static void main(String[] args) {
        // Initialize ImageJ
        new ImageJ();

        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║  NanoSIMS-EM Correlation Debug Runner                    ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();

        CorrelationWorkflowTest test = new CorrelationWorkflowTest();

        try {
            // Initialize test
            test.setUp();

            // Run specific test based on argument
            String testToRun = args.length > 0 ? args[0] : "full";

            switch (testToRun.toLowerCase()) {
                case "step1":
                    System.out.println("Running Step 1 test only...\n");
                    test.testStep1Only();
                    break;

                case "quality":
                    System.out.println("Running image quality test...\n");
                    test.testImageQuality();
                    break;

                case "metrics":
                    System.out.println("Running alignment metrics test...\n");
                    test.testAlignmentMetrics();
                    break;

                case "full":
                default:
                    System.out.println("Running full correlation workflow...\n");
                    test.testFullCorrelationWorkflow();
                    break;
            }

            System.out.println("\n╔═══════════════════════════════════════════════════════════╗");
            System.out.println("║  Debug Session Completed                                  ║");
            System.out.println("╚═══════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("\n❌ Debug session failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
