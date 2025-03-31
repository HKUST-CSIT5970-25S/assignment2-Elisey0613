package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram count using "pairs" approach
 */
public class BigramFrequencyPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyPairs.class);

    /*
     * Mapper: emits <bigram, 1>, where bigram = (leftWord, rightWord)
     */
    private static class MyMapper extends
        Mapper<LongWritable, Text, PairOfStrings, IntWritable> {

	    private static final IntWritable ONE = new IntWritable(1);
	    private static final PairOfStrings BIGRAM = new PairOfStrings();
	
	    @Override
	    public void map(LongWritable key, Text value, Context context)
	            throws IOException, InterruptedException {
	        String line = value.toString();
	        String[] words = line.trim().split("\\s+");
	        
	        if (words.length < 2) return; // Skip lines with single word
	        
	        // Emit each bigram with count 1
	        for (int i = 0; i < words.length - 1; i++) {
	            String word1 = words[i].replaceAll("[^a-zA-Z]", "").toLowerCase();
	            String word2 = words[i+1].replaceAll("[^a-zA-Z]", "").toLowerCase();
	            
	            if (!word1.isEmpty() && !word2.isEmpty()) {
	                BIGRAM.set(word1, word2);
	                context.write(BIGRAM, ONE);
	                
	                // Also emit the left word with empty right for total count
	                BIGRAM.set(word1, "");
	                context.write(BIGRAM, ONE);
	            }
	        }
	    }
	}

    /*
     * Reducer: compute bigram relative frequencies
     */
	private static class MyReducer extends
        Reducer<PairOfStrings, IntWritable, PairOfStrings, FloatWritable> {

	    private final static FloatWritable VALUE = new FloatWritable();
	    private float marginal = 0;
	    private String currentWord = null;
	    private boolean emittedMarginal = false;
	
	    @Override
	    public void reduce(PairOfStrings key, Iterable<IntWritable> values,
	            Context context) throws IOException, InterruptedException {
	        
	        String leftWord = key.getLeftElement();
	        String rightWord = key.getRightElement();
	        
	        // Calculate the sum of counts for this key
	        int sum = 0;
	        for (IntWritable value : values) {
	            sum += value.get();
	        }
	        
	        if (rightWord.isEmpty()) {
	            // This is the marginal count (total for left word)
	            currentWord = leftWord;
	            marginal = sum;
	            emittedMarginal = false;
	            
	            // Output the total count for the word first
	            context.write(new PairOfStrings(leftWord, ""), 
	                         new FloatWritable(marginal));
	            emittedMarginal = true;
	        } else {
	            // Make sure we've emitted the marginal count first
	            if (!emittedMarginal) {
	                context.write(new PairOfStrings(leftWord, ""), 
	                             new FloatWritable(marginal));
	                emittedMarginal = true;
	            }
	            
	            // Calculate relative frequency
	            float relativeFreq = sum / marginal;
	            VALUE.set(relativeFreq);
	            context.write(key, VALUE);
	        }
	    }
	}

    /*
     * Combiner: sum up the counts of bigrams locally
     */
    private static class MyCombiner extends
            Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values,
                           Context context) throws IOException, InterruptedException {
            int sum = 0;
	    for (IntWritable value : values) {
	        sum += value.get();
	    }
	    SUM.set(sum);
	    context.write(key, SUM);
        }
    }

    /*
     * Partitioner: ensures that keys with the same left element are shuffled to
     * the same reducer.
     */
    private static class MyPartitioner extends
            Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value,
                                int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE)
                    % numReduceTasks;
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public BigramFrequencyPairs() {
    }

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    /**
     * Runs this tool.
     */
    @SuppressWarnings({ "static-access" })
    public int run(String[] args) throws Exception {
        Options options = new Options();

        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();

        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: "
                    + exp.getMessage());
            return -1;
        }

        // Lack of arguments
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + BigramFrequencyPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // Create and configure a MapReduce job
        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyPairs.class.getSimpleName());
        job.setJarByClass(BigramFrequencyPairs.class);

        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(PairOfStrings.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        /*
         * A MapReduce program consists of four components: a mapper, a reducer,
         * an optional combiner, and an optional partitioner.
         */
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setPartitionerClass(MyPartitioner.class);
        job.setReducerClass(MyReducer.class);

        // Delete the output directory if it exists already.
        Path outputDir = new Path(outputPath);
        FileSystem.get(conf).delete(outputDir, true);

        // Time the program
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyPairs(), args);
    }
}
