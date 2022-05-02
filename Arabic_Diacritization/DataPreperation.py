import pickle as pkl
import gc

from constants import *


class DataPreperation:
    ''' This class will prepare data for FFNN models'''

    def __init__(self, data, check_point_file, is_checpoint_exist=False):
        ''' Load check point if exist, else process data '''

        self.check_point_file = check_point_file

        if is_checpoint_exist:
            self.Load_check_point()
        else:
            self.CHAR_2_IDX, self.IDX_2_CHAR = CHAR_IDX()
            self.process_data(data)
        
        self.description()
        gc.collect()


    def description(self):
        ''' Get data description '''

        # size
        print(f"Dataset size Y: {len(self.Y)}")
        print(f"Dataset size X: {len(self.X)}")

        # sample size
        print(f"sample size X: {len(self.X[0])}")
        print(f"sample size Y: {len(self.Y[0])}")

        # sample
        print(f"X sample:\n{self.X[0]}\nY sample:\n{self.Y[0]}\n")
        print("-" * 45)


    def Load_check_point(self):
        ''' Load processed data '''
        
        print("loading data ...")
        file_name_X = f"{PICKLE_LOCATION}/X_{self.check_point_file}.pickle"
        file_name_Y = f"{PICKLE_LOCATION}/Y_{self.check_point_file}.pickle"

        with open(file_name_X, 'rb') as fileX, open(file_name_Y, 'rb') as fileY:
            gc.disable()
            self.X = pkl.load(fileX)
            self.Y = pkl.load(fileY)
            gc.enable()

        print("loading data finished")

    def save_check_point(self):
        ''' Save processed data '''

        print("\nSaving data ...")
        file_name_X = f"{PICKLE_LOCATION}/X_{self.check_point_file}.pickle"
        file_name_Y = f"{PICKLE_LOCATION}/Y_{self.check_point_file}.pickle"

        with open(file_name_X, 'wb') as fileX, open(file_name_Y, 'wb') as fileY:
            pkl.dump(self.X, fileX)    
            pkl.dump(self.Y, fileY)

            # notify when finished
            print("Processed data is saved")


    def get_y(self, line, i, diacritics):
        ''' Get character class '''

        y = [0] * len( diacritics ) # 15
        
        if i + 1 < len(line) and line[i + 1] in diacritics:
            diac = line[i + 1]

            # some diacritics are composed of 2 HARAKATS check the next
            if i + 2 < len(line) and line[i + 2] in diacritics and diac + line[i + 2] in diacritics:
                diac += line[i + 2]
            
            # set class value to 1
            y[DIACRITICS_CLASS[diac]] = 1
        else:
            y[0] = 1 # no diacritic
        
        return y


    def get_x(self, line, i, diacritics):
        ''' Get character x vector '''
        x = []

        after = []  # 50 after current char
        before = [] # 50 before current char

        # after (+ including i) current character
        for i_after in range(i, len(line)):

            # if all 50 filled break
            if len(after) >= 50:
                break
            
            # if character store it
            if line[i_after] not in diacritics:
                after.append(line[i_after])
        after_missing = 50 - len(after)


        # before (- excluding i) current character
        for i_before in range(i - 1, -1, -1):

            # if all 50 filled break
            if len(before) >= 50:
                break

            # if character store it
            if line[i_before] not in diacritics:
                before.append(line[i_before])
        before_missing = 50 - len(before)

        # reverse        
        before = before[::-1]

        # store before and after into x
        x.extend([0] * before_missing)
        x.extend([0] * after_missing)
        for c in before:
            if c in self.CHAR_2_IDX: x.append(self.CHAR_2_IDX[c])
            else: x.append(0)

        for c in after:
            if c in self.CHAR_2_IDX: x.append(self.CHAR_2_IDX[c])
            else: x.append(0)

        return x


    def process_data(self, data):
        '''
            According to the paper 
            x = [<PAD>, <PAD>, .., ت, س, <PAD>..] of length 100,
            y = diacritic

            This function will process the data into this format and save when done
        '''

        self.X = []
        self.Y = []

        diacritics = list(DIACRITICS_CLASS.keys())
        j = 0
        for line in data:
            for i, char in enumerate(line):
                
                # if not arabic, character ignore
                if char not in ARABIC_CHAR:
                    continue

                # get the character's y (diacritic) and character's x vector
                y = self.get_y(line, i, diacritics)
                x = self.get_x(line, i, diacritics)
                
                self.Y.append(y)
                self.X.append(x)

            # show progress every 1000 line
            if j % 100 == 0:
                # clear screen
                print("\033c", end="")

                print("Processing data ...")
                full_bar = "▮" * (j // 1000)
                empty_bar = " " * ( len(data) // 1000 - (j // 1000) )
                percent = round( (j/len(data)) * 100, 2)
                print(f"{full_bar}{empty_bar}|{percent}%", end="")

            j = j + 1

        # save checkpoint
        self.save_check_point()