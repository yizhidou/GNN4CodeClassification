package fileWalker;

import java.io.File;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public abstract class SourceFileWalker
{
	protected final String DEFAULT_FILENAME_FILTER = "*.{c,cpp,h,cc,hpp,java}";

	public abstract void setFilenameFilter(String filter);

	/**
	 * Add a listener object that will be informed of all visited source files
	 * and directories.
	 */

	public abstract void addListener(SourceFileListener listener);

	/**
	 * Walk list of files and directory names and report them to listeners.
	 * 
	 * @param fileAndDirNames:
	 *            A list of file and/or directory names
	 */

	public void walk(String[] fileAndDirNames) throws IOException
	{
		// System.out.println("now let us see what is in fileAndDirNames");
		// for (String filename : fileAndDirNames){
		// 	System.out.println(filename);
		// }
		// System.exit(666);
		for (String filename : fileAndDirNames)
		{

			if (!pathIsAccessible(filename))
			{
				System.err.println("Warning: Skipping " + filename
						+ " because it is not accessible");
				continue;
			}
			try 
			{
				System.out.println(filename + " is going to on walk");
				walkExistingFileOrDirectory(filename);
			}
			catch (Exception err)
			{	
				//log_str = filename; // + " : "  + err.getMessage();
				// Files.write(Paths.get("/home/liux19/yizhidou/Dataset/MVDDataset/orignal_data/record_collections/joern_extraction_error_record.txt"), filename.getBytes(), StandardOpenOption.APPEND);
				// System.out.println("filename="+filename);
				System.out.println("stacktrace in SourceFileWalker:");
				err.printStackTrace();
				// continue;
			}
			
		}
	}

	protected abstract void walkExistingFileOrDirectory(String dirName)
			throws IOException;

	private boolean pathIsAccessible(String path)
	{
		File file = new File(path);
		if (!file.exists())
			return false;

		// TODO: add more checks, for example, do we have sufficient
		// permissions for access?

		return true;
	}

}