class EmbeddedMapping():
    def __init__(self, series):
       """
       Initializes an EmbeddedMapping object based on unique values in the given series.

       Parameters:
       - series: A pandas Series containing categorical values.
       """
       
       values = series.unique().tolist()
       values.sort()  

       # Create a dictionary mapping each unique value to an integer value plus 1.
       # This helps in creating embeddings where the integer values start from 1.
       self.embedding_dict = {value: int_value + 1 for int_value, value in enumerate(values)}
       
       # Store the total number of unique values plus 1 to account for unknown values.
       self.num_values = len(values) + 1

    def get_mapping(self, value):    
       """
       Retrieves the integer mapping for the given value.

       If the value is present in the embedding_dict, returns its corresponding integer.
       Otherwise, returns the integer representing unknown values (num_values).

       Parameters:
       - value: The categorical value to retrieve the mapping for.

       Returns:
       An integer representing the mapping for the given value.
       """
       
       if value in self.embedding_dict:
           return self.embedding_dict[value]
       else:
           return self.num_values
