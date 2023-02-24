import sys
data_dir=sys.argv[1] + "/"
h = open(data_dir+sys.argv[2])
refs = h.readlines()
f = open(data_dir+sys.argv[3])
users = f.readlines()
h = open(sys.argv[4])
hypos = h.readlines()
reqd_user = sys.argv[5]
lang='en'

if reqd_user == 'agent' or reqd_user=='system':
	lang='hi'
reference = []
for i in range(len(users)):
	if users[i].strip() == reqd_user:
		reference.append(refs[i])

print("Reference saved to", end=" ")
print(data_dir+reqd_user+"."+lang)
with open(data_dir+reqd_user+"."+lang, "w") as outfile:
    outfile.write("".join(reference))


preds = []
for i in range(len(users)):
	if users[i].strip() == reqd_user:
		preds.append(hypos[i])

print("Predictions saved to",end=" ")
print(sys.argv[4]+"."+reqd_user)
with open( sys.argv[4]+"."+reqd_user, "w") as outfile:
    outfile.write("".join(preds))
