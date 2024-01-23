


def default_evaluation_mechanisms():

    # mechanisms used in the evaluation

    default_mechanisms = {
        "GaussianMechanism-0.2-1e-09-1": {
            "mechanism": "GaussianMechanism",
            "name": "hare",
            "target_epsilon": 0.2,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.0011299216139600957,
                0.001318241882953445,
                0.0015065621519467943,
                0.0018832026899334928,
                0.0022598432279201915,
                0.0030131243038935885,
                0.0037664053798669855,
                0.004519686455840383,
                0.006026248607787177,
                0.012052497215574354,
                0.024104994431148708,
                0.048209988862297416,
                753.2810759733972,
                7532810.759733971,
            ],
        },
        "GaussianMechanism-0.05-1e-09-1": {
            "mechanism": "GaussianMechanism",
            "name": "mice",
            "target_epsilon": 0.05,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                7.838198820993648e-05,
                9.144565291159256e-05,
                0.00010450931761324865,
                0.0001306366470165608,
                0.00015676397641987296,
                0.0002090186352264973,
                0.0002612732940331216,
                0.00031352795283974593,
                0.0004180372704529946,
                0.0008360745409059892,
                0.0016721490818119784,
                0.0033442981636239567,
                52.25465880662432,
                522546.5880662432,
            ],
        },
        "GaussianMechanism-0.75-1e-09-0.25": {
            "mechanism": "GaussianMechanism",
            "name": "elephant",
            "target_epsilon": 0.75,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.0012046215579956804,
                0.0012046215579956804,
                0.0012046215579956804,
                0.0016105415553690262,
                0.0018135015540556992,
                0.0024268392653039594,
                0.00304469482166847,
                0.0036671295512739066,
                0.0049259880243681985,
                0.010157746799547595,
                0.021692303270566805,
                0.05062358655567231,
                9549.588259408023,
                95509743.76871194,
            ],
        },
        "GaussianMechanism-0.75-1e-09-1": {
            "mechanism": "GaussianMechanism",
            "name": "elephant",
            "target_epsilon": 0.75,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.014326461773250945,
                0.016714205402126103,
                0.01910194903100126,
                0.023877436288751577,
                0.02865292354650189,
                0.03820389806200252,
                0.04775487257750315,
                0.05730584709300378,
                0.07640779612400504,
                0.15281559224801008,
                0.30563118449602017,
                0.6112623689920403,
                9550.97451550063,
                95509745.1550063,
            ],
        },
        "GaussianMechanism-0.05-1e-09-0.25": {
            "mechanism": "GaussianMechanism",
            "name": "mice",
            "target_epsilon": 0.05,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                6.5321523468053755e-06,
                6.5321523468053755e-06,
                6.5321523468053755e-06,
                8.709664473703175e-06,
                9.798420537152075e-06,
                1.3064816750877709e-05,
                1.633134099725264e-05,
                1.9597993285658255e-05,
                2.6131682024699288e-05,
                5.2271560497570606e-05,
                0.00010457592565001285,
                0.00020928322104080927,
                50.86840271401754,
                522545.2017718887,
            ],
        },
        "GaussianMechanism-0.2-1e-09-0.25": {
            "mechanism": "GaussianMechanism",
            "name": "hare",
            "target_epsilon": 0.2,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                9.42266597060426e-05,
                9.42266597060426e-05,
                9.42266597060426e-05,
                0.00012566219481100852,
                0.00014137996236349148,
                0.00018855995041727747,
                0.00023576665156527232,
                0.0002830000935476473,
                0.0003775473111883151,
                0.0007568113925187969,
                0.0015205475118107293,
                0.003069265207061408,
                751.8948198807902,
                7532809.3734396165,
            ],
        },
        "LaplaceMechanism-0.1-1e-09-1": {
            "mechanism": "LaplaceMechanism",
            "name": "hare",
            "target_epsilon": 0.1,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.007247489719470035,
                0.008447868499758204,
                0.009644213591167239,
                0.012022085087405024,
                0.014375821154125196,
                0.01899122611681762,
                0.023454708121154175,
                0.02773704425216516,
                0.03567679319812044,
                0.0586645376180924,
                0.07820578888632694,
                0.08912223676169788,
                0.0999993372597735,
                0.10000003033783249,
            ],
        },
        "LaplaceMechanism-0.01-1e-09-0.25": {
            "mechanism": "LaplaceMechanism",
            "name": "mice",
            "target_epsilon": 0.01,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                6.229305218360959e-06,
                6.229305218360959e-06,
                6.229305218360959e-06,
                8.30575336319602e-06,
                9.343977435613549e-06,
                1.2458636889904154e-05,
                1.5573264204968296e-05,
                1.868784000622625e-05,
                2.4916759577291764e-05,
                4.9826504406791373e-05,
                9.958537641724564e-05,
                0.0001985868194859929,
                0.0025092189847730063,
                0.002509391048042071,
            ],
        },
        "LaplaceMechanism-0.01-1e-09-1": {
            "mechanism": "LaplaceMechanism",
            "name": "mice",
            "target_epsilon": 0.01,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                7.474972040732997e-05,
                8.720719340571392e-05,
                9.966422728308055e-05,
                0.00012457666785370414,
                0.0001494864226593562,
                0.00019929539894769643,
                0.0002490862088568302,
                0.00029885391455257703,
                0.00039830032487542244,
                0.0007941858392725987,
                0.0015691090964947535,
                0.002996410403694461,
                0.009999308639644067,
                0.010000001717703058,
            ],
        },
        "LaplaceMechanism-0.1-1e-09-0.25": {
            "mechanism": "LaplaceMechanism",
            "name": "hare",
            "target_epsilon": 0.1,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.0006054959555061024,
                0.0006054959555061024,
                0.0006054959555061024,
                0.0008074626183403785,
                0.0009084459497575165,
                0.0012112891990966879,
                0.001513845020312543,
                0.0018159337294593625,
                0.0024179994705545277,
                0.004773401856796052,
                0.009008632664587663,
                0.014918789521579068,
                0.02595286501068482,
                0.025953025646930906,
            ],
        },
        "LaplaceMechanism-0.25-1e-09-0.25": {
            "mechanism": "LaplaceMechanism",
            "name": "elephant",
            "target_epsilon": 0.25,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.0036305154012246377,
                0.0036305154012246377,
                0.0036305154012246377,
                0.0048461220107408704,
                0.005453925315498986,
                0.007274208769383579,
                0.009085010962763804,
                0.010880172403229648,
                0.014400552395930173,
                0.026926200728777175,
                0.042920612974223504,
                0.05516007067366637,
                0.06859857742970438,
                0.0685987165700904,
            ],
        },
        "LaplaceMechanism-0.25-1e-09-1": {
            "mechanism": "LaplaceMechanism",
            "name": "elephant",
            "target_epsilon": 0.25,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.04290381757089812,
                0.04979929086010807,
                0.05656351068351437,
                0.06962898531072477,
                0.08198516024643518,
                0.10429806177650719,
                0.12329143488411898,
                0.139153349259109,
                0.16310861812404373,
                0.20593365783043752,
                0.2281484035448891,
                0.23912213266486296,
                0.24999928083651946,
                0.24999997391457843,
            ],
        },
        "RandResponseMechanism-0.01-1e-09-0.25": {
            "mechanism": "RandResponseMechanism",
            "name": "mice",
            "target_epsilon": 0.01,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                6.242874841189128e-06,
                6.242874841189128e-06,
                6.242874841189128e-06,
                9.8847446710811e-06,
                1.1705679586027085e-05,
                1.768870561998555e-05,
                2.3777247082010233e-05,
                2.9732217122369064e-05,
                4.0811329591169555e-05,
                7.442349568485737e-05,
                0.00012548501913784408,
                0.00022417012478925273,
                0.0025077770887438258,
                0.0025079478974305803,
            ],
        },
        "RandResponseMechanism-0.25-1e-09-1": {
            "mechanism": "RandResponseMechanism",
            "name": "elephant",
            "target_epsilon": 0.25,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.04627322619855734,
                0.053679483294099585,
                0.0609291119126118,
                0.07487852032429054,
                0.08799138407154708,
                0.11142407678763588,
                0.13106212653768123,
                0.1472006609197566,
                0.17103844616481498,
                0.21162739615188372,
                0.23141603497354846,
                0.24085286306197556,
                0.249994218847408,
                0.2499947947320886,
            ],
        },
        "RandResponseMechanism-0.1-1e-09-1": {
            "mechanism": "RandResponseMechanism",
            "name": "hare",
            "target_epsilon": 0.1,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.007484858442661668,
                0.008724191916042715,
                0.00995916841070621,
                0.01241312071939027,
                0.01484102669998702,
                0.019597593912596984,
                0.024190682791846718,
                0.028589339396471503,
                0.03671865289559978,
                0.05998059582172366,
                0.07927517078425263,
                0.08977449655042088,
                0.10000229995556603,
                0.10000294428703238,
            ],
        },
        "RandResponseMechanism-0.01-1e-09-1": {
            "mechanism": "RandResponseMechanism",
            "name": "mice",
            "target_epsilon": 0.01,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                7.49125494845515e-05,
                8.739715602497829e-05,
                9.988132094185254e-05,
                0.00012484801431929723,
                0.00014981200662883554,
                0.0001997293979156082,
                0.0002496285193549219,
                0.0002995044050384976,
                0.0003991666689603538,
                0.0007959048219315478,
                0.0015724391236162274,
                0.003002313198740307,
                0.009993584087348736,
                0.009994272181750656,
            ],
        },
        "RandResponseMechanism-0.25-1e-09-0.25": {
            "mechanism": "RandResponseMechanism",
            "name": "elephant",
            "target_epsilon": 0.25,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.003918784647725451,
                0.003918784647725451,
                0.003918784647725451,
                0.00628111649723453,
                0.007462282421989069,
                0.011322110512156055,
                0.015181631537623114,
                0.018853441707624395,
                0.025307967840092958,
                0.040660778603671115,
                0.05267618536339245,
                0.060548096351138204,
                0.06859706023180004,
                0.06859716424742017,
            ],
        },
        "RandResponseMechanism-0.1-1e-09-0.25": {
            "mechanism": "RandResponseMechanism",
            "name": "hare",
            "target_epsilon": 0.1,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.00062536229351845,
                0.00062536229351845,
                0.00062536229351845,
                0.0009920936208073123,
                0.0011754592844517436,
                0.001777474302331809,
                0.0023884694029321085,
                0.002983493448630603,
                0.004080492665787837,
                0.007265401230156776,
                0.011408546037881289,
                0.0167110054699553,
                0.025953662612179965,
                0.025953810124721546,
            ],
        },
        "SVTLaplaceMechanism-0.1-1e-09-0.25": {
            "mechanism": "SVTLaplaceMechanism",
            "name": "hare",
            "target_epsilon": 0.1,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.0006253254869070662,
                0.0006253254869070662,
                0.0006253254869070662,
                0.000992035115614238,
                0.0011753899299678239,
                0.0017773693564881639,
                0.0023883284327268217,
                0.0029833175485207697,
                0.004080252960340853,
                0.007264984368909122,
                0.011407933039922229,
                0.016710233645566598,
                0.02595285761317631,
                0.025953018249425853,
            ],
        },
        "SVTLaplaceMechanism-0.1-1e-09-1": {
            "mechanism": "SVTLaplaceMechanism",
            "name": "hare",
            "target_epsilon": 0.1,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.007484419053572641,
                0.008723680249865797,
                0.009958584975923694,
                0.012412395606300962,
                0.0148401629134017,
                0.01959646406236845,
                0.024189305036559885,
                0.028587734606379003,
                0.036716661684110107,
                0.05997782328517301,
                0.07927219524548099,
                0.08977153296771603,
                0.09999930978165121,
                0.1000000028597102,
            ],
        },
        "SVTLaplaceMechanism-0.01-1e-09-0.25": {
            "mechanism": "SVTLaplaceMechanism",
            "name": "mice",
            "target_epsilon": 0.01,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                6.250035087154715e-06,
                6.250035087154715e-06,
                6.250035087154715e-06,
                9.89608216966457e-06,
                1.1719105710919497e-05,
                1.770899426920917e-05,
                2.380451909432535e-05,
                2.976631899991311e-05,
                4.0858137136913696e-05,
                7.450883246732747e-05,
                0.00012562878462331025,
                0.00022442613051287807,
                0.00250921904547007,
                0.0025093911087389073,
            ],
        },
        "SVTLaplaceMechanism-0.25-1e-09-1": {
            "mechanism": "SVTLaplaceMechanism",
            "name": "elephant",
            "target_epsilon": 0.25,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.04627512656530797,
                0.05368167551347873,
                0.060931583586390205,
                0.07488150855616496,
                0.08799482704236074,
                0.11142823905775412,
                0.13106677977117226,
                0.14720562849959237,
                0.17104371499373097,
                0.21163274292949238,
                0.23033626264670587,
                0.23920060523977202,
                0.2499993012786017,
                0.2499999943566607,
            ],
        },
        "SVTLaplaceMechanism-0.01-1e-09-1": {
            "mechanism": "SVTLaplaceMechanism",
            "name": "mice",
            "target_epsilon": 0.01,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                7.499846796235028e-05,
                8.749739237126041e-05,
                9.999587414287078e-05,
                0.00012499119747374533,
                0.00014998381353832985,
                0.00019995842803111402,
                0.0002499147307675029,
                0.00029984774446813287,
                0.0003996240593159781,
                0.0007968140506259152,
                0.001574213831662702,
                0.003005554972588877,
                0.009999308880619789,
                0.010000001958678778,
            ],
        },
        "SVTLaplaceMechanism-0.25-1e-09-0.25": {
            "mechanism": "SVTLaplaceMechanism",
            "name": "elephant",
            "target_epsilon": 0.25,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.003918948190341082,
                0.003918948190341082,
                0.003918948190341082,
                0.006281381830116163,
                0.0074625986500037045,
                0.011322592071194662,
                0.015182275459233091,
                0.018854235767543925,
                0.02530901077633369,
                0.040662279187107776,
                0.05267782763976633,
                0.060544314138235376,
                0.0685985835566911,
                0.06859872269707387,
            ],
        },
        "MLPateGaussianMechanism-0.75-1e-09-1": {
            "mechanism": "MLPateGaussianMechanism",
            "name": "elephant",
            "target_epsilon": 0.75,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.014326458475874517,
                0.016714201555186936,
                0.019101944634499355,
                0.023877430793124193,
                0.028652916951749034,
                0.03820388926899871,
                0.047754861586248386,
                0.05730583390349807,
                0.07640777853799742,
                0.15281555707599484,
                0.3056311141519897,
                0.6112622283039794,
                9550.972317249678,
                95509723.17249678,
            ],
        },
        "MLPateGaussianMechanism-0.2-1e-09-0.25": {
            "mechanism": "MLPateGaussianMechanism",
            "name": "hare",
            "target_epsilon": 0.2,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                9.42266526521296e-05,
                9.42266526521296e-05,
                9.42266526521296e-05,
                0.00012566218540172036,
                0.00014137995177651574,
                0.0001885599362932977,
                0.0002357666339003195,
                0.00028300007233774684,
                0.00037754728287632745,
                0.0007568113356370088,
                0.00152054739700228,
                0.003069264973157767,
                751.8947635286158,
                7532808.809917873,
            ],
        },
        "MLPateGaussianMechanism-0.2-1e-09-1": {
            "mechanism": "MLPateGaussianMechanism",
            "name": "hare",
            "target_epsilon": 0.2,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.0011299215294318342,
                0.0013182417843371399,
                0.0015065620392424456,
                0.001883202549053057,
                0.0022598430588636684,
                0.003013124078484891,
                0.003766405098106114,
                0.004519686117727337,
                0.006026248156969782,
                0.012052496313939565,
                0.02410499262787913,
                0.04820998525575826,
                753.2810196212228,
                7532810.196212227,
            ],
        },
        "MLPateGaussianMechanism-0.05-1e-09-0.25": {
            "mechanism": "MLPateGaussianMechanism",
            "name": "mice",
            "target_epsilon": 0.05,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                6.532152778335187e-06,
                6.532152778335187e-06,
                6.532152778335187e-06,
                8.709665048886595e-06,
                9.798421184162298e-06,
                1.3064817613613519e-05,
                1.6331342075792676e-05,
                1.959799457993405e-05,
                2.613168375052512e-05,
                5.227156395034941e-05,
                0.00010457593255989054,
                0.00020928323487798758,
                50.868406164570175,
                522545.236277415,
            ],
        },
        "MLPateGaussianMechanism-0.75-1e-09-0.25": {
            "mechanism": "MLPateGaussianMechanism",
            "name": "elephant",
            "target_epsilon": 0.75,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.001204621278252327,
                0.001204621278252327,
                0.001204621278252327,
                0.001610541180337599,
                0.0018135011313802352,
                0.0024268386976017222,
                0.0030446941068020955,
                0.0036671286870623154,
                0.0049259868547430075,
                0.010157744311449913,
                0.02169229758181141,
                0.0506235708340413,
                9549.586061157072,
                95509721.78620242,
            ],
        },
        "MLPateGaussianMechanism-0.05-1e-09-1": {
            "mechanism": "MLPateGaussianMechanism",
            "name": "mice",
            "target_epsilon": 0.05,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                7.838199338576543e-05,
                9.144565895005967e-05,
                0.0001045093245143539,
                0.0001306366556429424,
                0.00015676398677153086,
                0.0002090186490287078,
                0.0002612733112858848,
                0.0003135279735430617,
                0.0004180372980574156,
                0.0008360745961148312,
                0.0016721491922296625,
                0.003344298384459325,
                52.25466225717695,
                522546.6225717695,
            ],
        },
        "MLNoisySGDMechanism-0.75-1e-09-0.25": {
            "mechanism": "MLNoisySGDMechanism",
            "name": "elephant",
            "target_epsilon": 0.75,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                0.0006631053620001215,
                0.0006631053620001215,
                0.0006631053620001215,
                0.0010597533001708108,
                0.0012580772692561554,
                0.0019136685969850336,
                0.002583412219940004,
                0.0032403018725784215,
                0.004466764096536823,
                0.008262132806598139,
                361.1276374747046,
                6162.596750370193,
                176130023.61115593,
                1761353227504.517,
            ],
        },
        "MLNoisySGDMechanism-0.2-1e-09-1": {
            "mechanism": "MLNoisySGDMechanism",
            "name": "hare",
            "target_epsilon": 0.2,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.0012966269139109198,
                0.0012966269139109198,
                0.0012966269139109198,
                0.001729176264620383,
                0.0019454509399751147,
                0.00259461582802294,
                0.003244121941278701,
                0.0038939696429139605,
                0.005194691268995372,
                0.01041131443761904,
                0.020911112985060796,
                0.04218236148356724,
                25277143.63461099,
                252824413877.89316,
            ],
        },
        "MLNoisySGDMechanism-0.75-1e-09-1": {
            "mechanism": "MLNoisySGDMechanism",
            "name": "elephant",
            "target_epsilon": 0.75,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                0.010557279532871371,
                0.010557279532871371,
                0.010557279532871371,
                0.014101689762050364,
                0.01587389487663986,
                0.021216192401390627,
                0.026584452480642243,
                0.031978960895229266,
                0.04284789388608553,
                0.08743479449671374,
                171.0616189292071,
                5890.243094230957,
                176130024.997412,
                1761353227505.9033,
            ],
        },
        "MLNoisySGDMechanism-0.05-1e-09-0.25": {
            "mechanism": "MLNoisySGDMechanism",
            "name": "mice",
            "target_epsilon": 0.05,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                5.752576779982643e-06,
                5.752576779982643e-06,
                5.752576779982643e-06,
                9.108957149561725e-06,
                1.0787147334351266e-05,
                1.6301480732653733e-05,
                2.1913392393851927e-05,
                2.7402548215493817e-05,
                3.761623201373584e-05,
                6.862149289067846e-05,
                0.00011583582323244061,
                0.00020774316056054466,
                1832065.5192280961,
                18373646585.238857,
            ],
        },
        "MLNoisySGDMechanism-0.05-1e-09-1": {
            "mechanism": "MLNoisySGDMechanism",
            "name": "mice",
            "target_epsilon": 0.05,
            "target_delta": 1e-09,
            "subsampling_prob": 1,
            "rdp_eps_cost": [
                9.203725765888414e-05,
                9.203725765888414e-05,
                9.203725765888414e-05,
                0.00012271803137126397,
                0.0001380584182274539,
                0.00018408126655784272,
                0.00023010580318204147,
                0.0002761320278099502,
                0.000368189542365358,
                0.0007364871527926371,
                0.0014734068316699302,
                0.002948545801445017,
                1832066.9054841888,
                18373646586.625153,
            ],
        },
        "MLNoisySGDMechanism-0.2-1e-09-0.25": {
            "mechanism": "MLNoisySGDMechanism",
            "name": "hare",
            "target_epsilon": 0.2,
            "target_delta": 1e-09,
            "subsampling_prob": 0.25,
            "rdp_eps_cost": [
                8.108845586922431e-05,
                8.108845586922431e-05,
                8.108845586922431e-05,
                0.0001285319018081201,
                0.00015225362477756799,
                0.00023025243501599105,
                0.0003096647048644735,
                0.0003873626792745277,
                0.0005319863217139176,
                0.000971864703738549,
                0.0016474877594338576,
                0.00298262829971025,
                25277142.248354893,
                252824413876.50693,
            ],
        },
    }

    default_mechanisms = dict(sorted(default_mechanisms.items()))
    return default_mechanisms