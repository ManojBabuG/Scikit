from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = ["Apple", "Apple", "Orange", "Orange"]

cls = tree.DecisionTreeClassifier()
clf = cls.fit(features, labels)

print(clf.predict([[150, 0]]))