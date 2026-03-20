import matplotlib.pyplot as plt
import seaborn as sns


def plot_pass_distribution(df):

    sns.countplot(x="pass", data=df)

    plt.title("Pass vs Fail")

    plt.show()


def plot_absences_vs_grade(df):

    sns.scatterplot(x="absences", y="G3", data=df)

    plt.title("Absences vs Final Grade")

    plt.show()