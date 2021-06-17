package tools.parser;

import java.io.IOException;

import org.apache.commons.cli.ParseException;
import java.lang.Exception;
import java.nio.file.Files;
import java.nio.file.Paths;

import fileWalker.OrderedWalker;
import fileWalker.SourceFileWalker;
import outputModules.parser.Parser;

/**
 * Main Class for the parser: This class processes command line arguments and
 * configures the parser in accordance. It then uses a SourceFileWalker to visit
 * source-files and directories and report them to the parser.
 */

public class ParserMain
{

	private static ParserCmdLineInterface cmd = new ParserCmdLineInterface();
	private static SourceFileWalker sourceFileWalker = new OrderedWalker();

	private static Parser parser;

	public static void main(String[] args)
	{
	    // try
	    // {
            parseCommandLine(args);
            String[] fileAndDirNames = getFileAndDirNamesFromCommandLine();
            setupIndexer();
            walkCodebase(fileAndDirNames);
            System.out.println("this is the end of main func");
        // } catch(Exception e){
            // System.out.println("Exception occurred");
            //Files.write(Paths.get("/home/liux19/yizhidou/Dataset/MVDDataset/orignal_data/record_collections/joern_extraction_error_record.txt"), fileAndDirNames.getBytes(), StandardOpenOption.APPEND);
		// }

	}

	private static void parseCommandLine(String[] args)
	{
		try
		{
			cmd.parseCommandLine(args);
		} catch (RuntimeException | ParseException ex)
		{
			printHelpAndTerminate(ex);
		}
	}

	private static void printHelpAndTerminate(Exception ex)
	{
		System.err.println(ex.getMessage());
		cmd.printHelp();
		System.exit(1);
	}

	private static String[] getFileAndDirNamesFromCommandLine()
	{
		return cmd.getFilenames();
	}

	private static void setupIndexer()
	{

		String outputFormat = cmd.getOutputFormat();
		if (outputFormat.equals("neo4j"))
			parser = new CParserNeo4JOuput();
		else if (outputFormat.equals("csv"))
			parser = new CParserCSVOutput();
		else
			throw new RuntimeException("unknown output format");

		String outputDir = cmd.getOutputDir();
		System.out.println("outputDir:");
		System.out.println(outputDir);
		parser.setOutputDir(outputDir);
		parser.initialize();
		sourceFileWalker.addListener(parser);
	}

	private static void walkCodebase(String[] fileAndDirNames)
	{
		try
		{
			sourceFileWalker.walk(fileAndDirNames);
		} catch (Exception err)  //IOException
		{
			System.err
					.println("Error walking source files: " + err.getMessage());
		}
		//finally
		//{
		    //System.err
					//.println("Error walking source files: " + err.getMessage());
			//parser.shutdown();
		//}
	}
}
