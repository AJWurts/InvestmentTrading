    # # print(bins)

    # print(bins.bin.value_counts())
    # print(bins)


    # buying = bins[bins.bin == 1.0]
    # selling = bins[bins.bin == -1]

    # ax = plt.gca()
    
    # buyingX = [i for i in buying.index]
    # buyingY = [data['Close'][x] for x in buying.index]
    # buyingSize = [abs(ret) * 1000 for ret in buying['ret']]


    # plt.scatter(buyingX, buyingY, color='green', zorder=3, s=buyingSize)

    # sellingX = [i for i in selling.index]
    # sellingY = [data['Close'][x] for x in selling.index]
    # sellingSize = [abs(ret) * 1000 for ret in selling['ret']]

    # plt.scatter(sellingX, sellingY, color='red', zorder=2, s=sellingSize)


    # data.plot(y='Close', ax = ax, zorder=1)

    # plt.show()