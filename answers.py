a1aa=['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
a1b=2649
a1c=12.061525956381585
a1d='function'
a2a=13
a2b=2.4630366660416674
a3c=2.883894963080882
a3d='DET'
a4a3=0.8941143860297515
a4b1=[("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'CONJ'), ('racing', 'ADJ'), ('cars', 'NOUN'), ('.', '.')]
a4b2=[("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'ADP'), ('racing', 'VERB'), ('cars', 'NOUN'), ('.', '.')]
a4b3='Here we see the algorithm understanding sentence structure,but not context.Though to us it\'s clear it is one\nphrase,it works grammatically to have two phrases:"Im useless for anything" and "racing cars",conjoined by a\nbut.And it may have seen racing cars ADJ NOUN in training'
a4c=60.73342283709657
a4d=70.21922688308214
a4e=['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB', 'ADV']
a5t0=0.558440358495796
a5tk=0.6090732698882011
a5b='1:Labeled data has no cases of he,but unlabelled does,so tk can learn the tag which makes the most sense\nstructurally,ie PRON\n\n2:t0 tags them as PROUN,but tk tags it NOUN.tk must have found a case where its more likely a noun.As nouns\nare more common,the prob its noun increases,and each time the data is tagged,more ones are tagged as a\nnoun,increasing the prob more.This shows EM working badly as a wrong tag is used to tag unlabelled data,creating\nmore wrong tags and reinforcing that tag'
a6='This could be done tagging each word,then using PCFG to create a constituent tree from the tags.This will create a\nparse(as long as POS is mostly correct),but not always the right one as words have many usages.Like 4b even if\ngrammatical,still incorrect,without context(aided by lexical coverage)\n\nBut the less lexical coverage PCFG has,the more it benefits from POS tagging.But if POS tagger is bad,and coverage\nis still high,it will have a negative affect,creating conflicting sentence structures'
a7="The Brown tagset contains a large amount of very specific grammatical tags.This means the algorithm will need lots\nof examples of each tag's usage to create an accurate tagger.And more tags mean less examples in the same amount of\ndata,and more uncommon tags,so the Brown tagset would create a less accurate model,and worse when using EM,as you\nstart with a very small sample size.To get a similar accuracy to Universal tags,you'd need a larger dataset,with\nan equivalent amount of examples of tags"
a4full_vit=[[26.782208935674642, 27.175612695180405, 26.83747232781355, 25.69387747486122, 26.069930312266184, 25.33122669020975, 26.602772790690192, 26.7617409338053, 6.7157647918435055, 26.154747513228394, 28.346102791874564, 27.06281489027102], [34.01526253042612, 26.139704758753695, 34.54383920395947, 32.31894404945449, 34.25048137679512, 36.22833529220789, 27.617716596488677, 37.174460980126284, 34.96569792142534, 33.2167214529358, 30.719087535156927, 43.87771626842389], [52.96761801645391, 52.61690686578734, 53.24942252459456, 54.723565975137475, 52.53968502182914, 34.01218757982337, 51.358458806319945, 52.70225028680407, 55.14157828038837, 53.01419324341059, 54.12151433519213, 54.72321304758704], [64.02174333349448, 58.59624615156186, 64.37034195590547, 61.740806181353804, 67.52308363114037, 64.95730590909109, 44.113599640606985, 60.603762131928086, 62.65031911857109, 64.14971027238593, 61.85439770355394, 60.37766863378028]]
a4full_bp=[['PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON'], ['NOUN', 'ADJ', 'NOUN', 'NOUN', 'ADJ', 'VERB', 'ADJ', 'ADJ', 'NOUN', 'ADJ', 'NOUN', 'ADJ'], ['DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET']]
