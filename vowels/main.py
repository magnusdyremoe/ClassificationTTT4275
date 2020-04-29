import vowels_task1 as vt1
import vowels_task2 as vt2
import latexconfusiontable as lct


def main():
    print("Task 1")
    #conf1_train, conf1_test = vt1.train_test_single_gaussian(0,70,False)
    #lct.print_confusion(conf1_train)
    #lct.print_confusion(conf1_test)
    print("\n______________________________\n")

    print("Task 2, 2 gaussians")
    gmm2_train, gmm2_test = vt2.train_test_GMM(0,70,2)
    #lct.print_confusion(gmm2_train)
    #lct.print_confusion(gmm2_test)

    print("\n_______________________________\n")

    print("Task 2, 3 gaussians")
    gmm3_train, gmm3_test = vt2.train_test_GMM(0,70,3)
    #lct.print_confusion(gmm3_train)
    #lct.print_confusion(gmm3_test)


main()