from pnl_randomise_utils import np, pd
from pnl_randomise_utils import print_head, print_df


class RandomiseConMat(object):
    def get_contrast_info(self):
        """Read design contrast file into a numpy array

        self.contrast_array : numpy array of the contrast file excluding the
                           headers
        """

        with open(self.contrast_file, 'r') as f:
            lines = f.readlines()
            headers = [x for x in lines
                       if x.startswith('/')]

        last_header_line_number = lines.index(headers[-1]) + 1
        self.contrast_array = np.loadtxt(self.contrast_file,
                                         skiprows=last_header_line_number)
        self.contrast_df = pd.DataFrame(self.contrast_array)
        self.contrast_df.columns = [f'col {int(x)+1}' for x in
                                    self.contrast_df.columns]

    def get_contrast_info_english(self):
        """Read design contrast file into a numpy array

        self.contrast_lines : attributes that states what each row in the
                           contrast array represents
       TODO:
           select group columns
        """

        # if all lines are group comparisons -->
        # simple group comparisons or interaction effect
        if (self.contrast_array.sum(axis=1) == 0).all():
            # TODO : do below at the array level?
            df_tmp = pd.DataFrame(self.contrast_array)
            self.contrast_lines = []
            for contrast_num, row in df_tmp.iterrows():
                # name of the column with value of 1
                pos_col_num = row[row == 1].index.values[0]
                neg_col_num = row[row == -1].index.values[0]

                # if interaction : they have zeros in the group column
                # TODO : make this more efficient later
                half_cols = (self.contrast_array.shape[0] / 2) + 1
                if pos_col_num not in list(range(int(half_cols+1))):
                    if pos_col_num < neg_col_num:
                        text = 'Negative Interaction'
                    else:
                        text = 'Positive Interaction'

                else:
                    # Change order of columns according to their column numbers
                    if pos_col_num < neg_col_num:
                        if self.group_labels:
                            text = f'{self.group_labels[pos_col_num]} > {self.group_labels[neg_col_num]}'
                        else:
                            text = f'Group {pos_col_num+1} > Group {neg_col_num+1}'
                    else:
                        if self.group_labels:
                            text = f'{self.group_labels[neg_col_num]} < {self.group_labels[pos_col_num]}'
                        else:
                            text = f'Group {neg_col_num+1} < Group {pos_col_num+1}'
                self.contrast_lines.append(text)

        # if all rows sum to 1 --> the correlation contrast
        # TODO: there is a positibility of having 0.5 0.5?
        elif (np.absolute(self.contrast_array.sum(axis=1)) == 1).all():
            # TODO : do below at the array level?
            df_tmp = pd.DataFrame(self.contrast_array)
            self.contrast_lines = []
            for contrast_num, row in df_tmp.iterrows():
                # name of the column with value of 1
                col_num = row[row != 0].index.values[0]

                # Change order of columns according to their column numbers
                if row.loc[col_num] == 1:
                    text = f'Positively correlated with col {col_num+1}'
                else:
                    text = f'Negatively correlated with col {col_num+1}'
                self.contrast_lines.append(text)

        # TODO add interaction information
        # if group column is zero
        # 0   0   1    -1
        try:
            if np.equal(np.array(self.contrast_array), 
                        np.array([[0, 0, 1, -1], [0, 0, -1, 1]])).all():
                self.contrast_lines = ['Positive Interaction', 
                                       'Negative Interaction']
        except ValueError:
            pass


    def get_matrix_info(self):
        """Read design matrix file into a numpy array and summarize

        matrix_header: headers in the matrix file
        matrix_array: numpy array matrix part of the matrix file
        matrix_df: pandas dataframe of matrix array

        TODO:
            add function defining which column is group
        """
        with open(self.matrix_file, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            matrix_lines = [x for x in lines if x.startswith('/')]

        self.matrix_header = [
            x for x in matrix_lines if
            not x.startswith('/NumWaves') and
            not x.startswith('/NumContrasts') and
            not x.startswith('/NumPoints') and
            not x.startswith('/PPheights') and
            not x.startswith('/Matrix')]

        self.matrix_header = '\n'.join(
            [x[1:].strip() for x in self.matrix_header])

        the_line_with_matrix = lines.index('/Matrix') + 1
        self.matrix_array = np.loadtxt(self.matrix_file,
                                       skiprows=the_line_with_matrix)

        # matrix array into pandas dataframe
        self.matrix_df = pd.DataFrame(self.matrix_array)
        # rename columns to have 'col ' in each
        self.matrix_df.columns = [f'col {x+1}' for x in
                                  self.matrix_df.columns]

        # summarize matrix
        self.matrix_info = self.matrix_df.describe()
        self.matrix_info = self.matrix_info.loc[
                ['mean', 'std', 'min', 'max'], :]
        self.matrix_info = self.matrix_info.round(decimals=2)

        # For each column of the matrix, add counts of unique values to
        # self.matrix_info
        for col in self.matrix_df.columns:
            # create a dataframe that contains unique values as the index
            unique_values = self.matrix_df[col].value_counts().sort_index()
            # If there are less unique values than the half the number of
            # all data  for the column
            if len(unique_values) < len(self.matrix_df) / 2:
                # unique values as an extra row
                self.matrix_info.loc['unique values', col] = np.array2string(
                    unique_values.index)[1:-1]
                # count of each unique value as an extra row
                self.matrix_info.loc['count', col] = np.array2string(
                    unique_values.values)[1:-1]
            # If there are 5 or more unique values in the column, leave the
            else:
                # 'unique' and 'count' as 'continuous values'
                self.matrix_info.loc['unique values', col] = \
                        'continuous values'
                self.matrix_info.loc['count', col] = 'continuous values'

        # define which column represent group column
        # Among columns where their min value is 0 and max value is 1,
        min_0_max_1_col = [x for x in self.matrix_df.columns
                           if self.matrix_df[x].isin([0, 1]).all()]

        if 'col 1' not in min_0_max_1_col:
            print("Contrast file does not have 1s in the first column "
                  "- likely be correlation or interaction")
            print("Setting self.group_cols=['no group col']")
            self.group_cols = ['no group col']

        else:
            # if sum of each row equal to 1, these columns would highly likely
            # be group columns
            if (self.matrix_df[min_0_max_1_col].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col
            # If not, remove a column from the list of columns at the end, and
            # test whether each row sums to 1
            elif (self.matrix_df[min_0_max_1_col[:-1]].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col[:-1]
            elif (self.matrix_df[min_0_max_1_col[:-2]].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col[:-2]
            elif (self.matrix_df[min_0_max_1_col[:-3]].sum(axis=1) == 1).all():
                self.group_cols = min_0_max_1_col[:-3]
            else:
                self.group_cols = min_0_max_1_col[0]

            # 'unique' and 'count' columns of group columns
            for group_num, col in enumerate(self.group_cols, 1):
                # unique values as an extra row
                # self.matrix_info.loc['column info', col] = f"Group {group_num}"
                if self.group_labels != False:
                    self.matrix_info.loc['column info', col] = \
                        self.group_labels[group_num-1]
                else:
                    self.matrix_info.loc['column info', col] = \
                        f"Group {group_num}"

                # count of each unique value as an extra row
                self.matrix_info.loc['count', col] = \
                    (self.matrix_df[col] == 1).sum()

            # wide to long
            df_tmp = self.matrix_df.copy()
            # print_df(df_tmp)
            for num, row in df_tmp.iterrows():
                for group_num, group_col in enumerate(self.group_cols, 1):
                    if row[group_col] == 1:
                        if self.group_labels != False:
                            df_tmp.loc[num, 'group'] = f'{self.group_labels[group_num-1]}'
                        else:
                            df_tmp.loc[num, 'group'] = f'Group {group_num}'
            # print_df(df_tmp)
                
            # # non-group column
            self.covar_info_dict = {}
            for col in self.matrix_info.columns:
                if col not in self.group_cols:
                    unique_values = self.matrix_df[col].unique()
                    if len(unique_values) < 5:
                        count_df_tmp = df_tmp.groupby(['group', col]).count()
                        count_df_tmp = count_df_tmp[['col 1']]
                        count_df_tmp.columns = ['Count']
                        self.covar_info_dict[col] = \
                            count_df_tmp.reset_index().set_index('group')
                    else:
                        self.covar_info_dict[col] = \
                            df_tmp.groupby('group').describe()[col]

    def print_matrix_info(self):
        print_head('Matrix summary')
        print(f'Contrast file : {self.contrast_file}')
        print(f'Matrix file : {self.matrix_file}')
        print()
        if hasattr(self, 'matrix_df'):
            print(f'total number of data point : {len(self.matrix_df)}')
        if hasattr(self, 'group_cols'):
            print(f'Group columns are : ' + ', '.join(self.group_cols))
        if hasattr(self, 'matrix_info'):
            print_df(self.matrix_info)
