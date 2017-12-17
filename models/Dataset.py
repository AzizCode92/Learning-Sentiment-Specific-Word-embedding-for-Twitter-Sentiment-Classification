import csv
import sys
import random
from Review import Review
from sklearn.model_selection import train_test_split

class DatasetReview():
    """docstring for Dataset"""

    def __init__(self):
        self.dataset = []
        self.field_names = []
        self.label_values = []
        self.column_label = ""

    def load_review_from_csv(self, infile):
        with open(infile, "rb") as csvfile:
            reader = csv.DictReader(csvfile)

            # init field names & label column
            self.field_names = reader.fieldnames
            self.column_label = self.field_names[-1]

            for rows in reader:
                review = Review(rows[self.field_names[0]], rows[self.field_names[1]])
                self.dataset.append(review)
                if self.label_values.count(rows[self.column_label]) == 0:
                    self.label_values.append(rows[self.column_label])

        return infile

    def dataset_from_array(self, dataset):
        n_dataset = DatasetReview()
        n_dataset.dataset = dataset
        n_dataset.field_names = self.field_names
        n_dataset.label_values = self.label_values
        n_dataset.column_label = self.column_label

        return n_dataset

    def dataset_from_contents_labels(self, contents, labels):
        arr_dataset = []
        for i in xrange(len(contents)):
            dr = Review(contents[i], labels[i])
            arr_dataset.append(dr)

        return self.dataset_from_array(arr_dataset)

    def get_dataset_size(self):
        return len(self.dataset)

    """get text content for datasets"""

    def get_contents(self):
        res = []
        for data in self.dataset:
            res.append(data.content)

        return res

    """ get labels for all datasets """

    def get_labels(self):
        res = []
        for data in self.dataset:
            res.append(data.polarity)

        return res

    def get_label_enum(self):
        return self.label_values

    def get_dataset(self, idx):
        return self.dataset[idx]

    def get_formatted_dataset(self):
        res = []
        for data in self.dataset:
            res.append(data.to_string())

        return res

    def export_formatted_dataset(self, outfile):
        res = self.get_formatted_dataset()
        with open(outfile, "wb") as f:
            for row in res:
                f.write(row + "\n")

        return outfile

    def export_to_csv(self, outfile):
        with open(outfile, "wb") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            writer.writeheader()
            for data in self.dataset:
                writer.writerow({
                    'content': data.content,
                    'polarity': data.polarity
                })

        return outfile

    def get_data_label_size(self, label):
        return sum(1 for x in self.dataset if x.polarity == label)

    def get_data_label(self, label):
        return [data for data in self.dataset if data.polarity == label]

    def get_sample_to_minority(self):
        if not self.dataset:
            return []
        else:
            pos_sample = self.get_data_label_size("positive")
            neg_sample = self.get_data_label_size("negative")
            neu_sample = self.get_data_label_size("neutral")

            print "%d | %d | %d" % (pos_sample, neg_sample, neu_sample)
            t_dataset = []
            if pos_sample > neg_sample:
                temp = self.get_data_label("positive")
                for x in xrange(0, neg_sample):
                    idx = random.randint(0, len(temp) - 1)
                    t_dataset.append(temp[idx])

                # append the minority instance
                t_dataset.extend(self.get_data_label("negative"))
                m_dts = self.dataset_from_array(t_dataset)
                return m_dts

            elif neg_sample > pos_sample:
                temp = self.get_data_label("negative")
                for x in xrange(1, pos_sample):
                    idx = random.randint(0, len(temp) - 1)
                    t_dataset.append(temp[idx])

                # append the minority instance
                t_dataset.extend(self.get_data_label("positive"))
                m_dts = self.dataset_from_array(t_dataset)
                return m_dts

            else:
                return self

    def split_to_ratio(self, ratio):
        X_train, X_test, y_train, y_test = train_test_split(self.get_contents(), self.get_labels(), test_size=ratio)

        dataset_train = self.dataset_from_contents_labels(X_train, y_train)
        dataset_test = self.dataset_from_contents_labels(X_test, y_test)

        return dataset_train, dataset_test

    def export_only_contents(self, outfile):
        with open(outfile, "wb") as ofile:
            for data in self.dataset:
                ofile.write(data.content + "\n")

        return outfile


def main(infile):
    dataset = DatasetReview()
    dataset.load_review_from_csv(infile)
    print dataset.get_label_enum()
    dataset.export_formatted_dataset("formatted_dataset.txt")

    print "Positive instances: %d" % (dataset.get_data_label_size("positive"))
    print "Negative instances: %d" % (dataset.get_data_label_size("negative"))

    t_dataset = dataset.get_sample_to_minority()
    print "Positive instances: %d" % (t_dataset.get_data_label_size("positive"))
    print "Negative instances: %d" % (t_dataset.get_data_label_size("negative"))
    t_dataset.export_to_csv("sample.csv")


if __name__ == '__main__':
    main(sys.argv[1])
