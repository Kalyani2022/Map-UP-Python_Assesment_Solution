# Map-UP-Python_Assesment_Solution

I have completed botht assesment of the Python and Excel here in my my repository i have created a Submissions folder in that i have added my Python_Task -1 Solution, Python_Task -2 Solution and Excel assesment file solution.

Here are Some points related to my Excel Assesment which i want to mention.

Instructions which are given to me perform the Excel assesment.

1. There are 5 tasks in total for the Excel assessment.	
2. Each task holds different weightage.	
3. For Task_1 and Task_2, please refer to the "Datasheet" tab.	
4. Please provide explanation to your solution wherever you're prompted to do so.

Allocation of marks by task	- This is weightage of Each Tasks.
Task	Marks
Task_1	25
Task_2	30
Task_3	10
Task_4	20
Task_5	40
Total	125

# Task -1 Questions which i have to perform.
Please carry out the following tasks within the "Dataset" tab.	
1	Highlight all '-1' values in red colour and all '0' values in green.	Red, Green
2	Freeze the top row	
3	Give the sum of all values greater than '20' in column H labelled 'car2'	0 -  This formula i have used to solve this qustion .   =SUMIF(H2:H40, ">20")
4	Format the dates given in column X labelled 'dates' in format: 'YYYY-MM-DD'	  


# Task  - 2 
1. Using relevant Excel formulas, fill up the corresponding empty cells in Table_A with values from the "Dataset" worksheet.						
2. Calculate minimum, maximum and average values for each column at the bottom.

# Task - 3 

1 Using either Excel formulas or built-in functionality, split the values given in Table_A into distinct cells as given in Example_table.	

# Formula and Functionality Used - Text Parsing or Text Splitting 

# Explanation:	"Step 1) - Select the cell or column that contains the values you want to split.
Step 2) - Go to the ""Data"" tab in the Excel ribbon.
Step 3) - Click on ""Text to Columns.""
Step 4) - Choose ""Delimited"" and click ""Next.""
Step 5) - Choose the delimiter (such as comma, space, or a custom delimiter) and click ""Next.""
Step 6) - Specify the destination for the split data and click ""Finish.""

# Task - 4
Using appropriate formula, calculate the sum of sales in Table_A using the following condition:										
All #N/A values must be substituted with the value '20'		

# Formula/Functionality used:	This Functionality we called it as Conditional Sum With Replacement.	

# Explanation:	"1) Identifies #N/A values: Checks the ""Sales"" column (C12 to C21) for #N/A values using ISNA.
2) Replaces #N/A with 20: Using IF, replaces any #N/A values with '20' and keeps non-#N/A values unchanged.
3) Calculates the sum: Computes the sum of the modified values in the ""Sales"" column using SUM.
4) Result: Provides the total sales, where any #N/A values have been substituted with '20'.

 Formula Used - =SUM(IF(ISNA(C12:C21), 20, C12:C21))

# Task -5 Fill the values in column "value" of Table_C using the tables A and B.								

# Formula/Functionality used:		We can called this functionality two - dimensional lookup or matrix lookup.	

# Explanation:	
"In Table_C, the formula =INDEX(Table_A, MATCH(A2, Table_B[ID], 0), MATCH(B2, Table_B[ID], 0)) uses the MATCH function to find the row and column indices in Table_A based on ""name_1"" and ""name_2."" The INDEX function then retrieves the corresponding value from Table_A. This process is repeated for each row in Table_C, dynamically populating the ""value"" column based on relationships specified in Table_B.												



"											
													

