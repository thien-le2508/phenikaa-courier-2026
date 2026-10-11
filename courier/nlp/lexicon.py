"""Từ điển cho bộ sinh câu yêu cầu tổng hợp.

Mỗi mục là danh sách cụm từ. Các cụm trong khóa "val" là cách diễn đạt đã biết là chỉ xuất hiện
trong validation; khi đo khả năng tổng quát hóa (holdout) ta loại chúng ra để điểm validation trung thực.
Không có cụm nào lấy từ dữ liệu test.
"""

ALIASES = {
    "library": {
        "train": ["thư viện", "thư viện trường", "khu đọc sách", "chỗ trả sách", "nơi mượn giáo trình", "phòng đọc"],
        "new": ["phòng đọc sách", "kho sách", "trung tâm học liệu", "trung tâm thông tin thư viện", "quầy mượn trả sách",
                "chỗ mượn sách", "phòng tra cứu", "khu tự học", "phòng đọc tự chọn", "kho giáo trình",
                "nơi trả giáo trình", "thư quán", "phòng sách", "khu mượn sách", "thư viện số", "kho lưu trữ sách", "phòng đọc báo", "thư viện trung tâm", "phòng mượn sách", "khu sách"],
        "val": ["nơi mượn sách tham khảo", "phòng tài liệu"],
    },
    "dorm": {
        "train": ["ký túc xá", "ký túc", "KTX", "khu KTX", "khu nội trú", "phòng ở sinh viên", "tòa nhà ở của sinh viên"],
        "new": ["nhà ở sinh viên", "khu ký túc", "tòa KTX", "khu lưu trú sinh viên", "nhà nội trú", "dãy nhà ở",
                "chỗ ở của sinh viên", "khu nhà ở sinh viên", "tòa ký túc", "khu ở nội trú", "khu lưu xá", "khu cư xá", "ký túc xá sinh viên", "nhà ký túc", "dãy ký túc", "khu ký túc xá", "chỗ ngủ của sinh viên", "tòa nội trú"],
        "val": ["dãy phòng nội trú", "khu ở của sinh viên"],
    },
    "sports": {
        "train": ["nhà thể chất", "nhà thi đấu", "khu thể thao", "sân thể thao", "sân bóng", "sân tập"],
        "new": ["nhà thể thao", "sân vận động", "phòng gym", "nhà tập đa năng", "khu thể dục", "sân điền kinh",
                "sân bóng rổ", "nhà tập thể thao", "sân cỏ", "bể bơi của trường", "khu tập luyện", "sân bóng đá", "phòng tập", "nhà tập", "khu tập thể dục", "sân chơi thể thao", "nhà thi đấu đa năng", "sân bóng chuyền", "khu rèn luyện thể chất"],
        "val": ["sân luyện tập", "khu vận động"],
    },
    "clinic": {
        "train": ["trạm y tế", "khu y tế", "phòng y tế", "phòng sơ cứu", "phòng khám của trường", "trạm xá"],
        "new": ["phòng khám", "trạm y tế trường", "cơ sở y tế", "nơi sơ cứu", "phòng cấp cứu", "phòng y tế trường",
                "phòng khám đa khoa của trường", "trung tâm y tế", "phòng điều trị", "góc sơ cứu", "phòng khám bệnh", "y tế trường", "trạm sơ cứu", "bệnh xá", "phòng thuốc", "phòng y", "khu sơ cứu", "phòng chăm sóc y tế"],
        "val": ["nơi khám sức khỏe", "phòng bác sĩ"],
    },
    "canteen": {
        "train": ["căn tin", "căng tin", "canteen", "nhà ăn", "bếp ăn", "khu ăn uống"],
        "new": ["nhà ăn sinh viên", "khu ẩm thực", "căng-tin", "quầy cơm", "nhà bếp", "khu bếp", "quán ăn của trường",
                "căn-tin", "food court", "bếp tập thể", "quầy ăn uống", "bếp cơm", "quán cơm sinh viên", "khu căng tin", "quầy bán đồ ăn", "khu ăn", "căng tin trường", "quán ăn sinh viên", "nhà bếp trung tâm"],
        "val": ["nơi phục vụ bữa trưa", "phòng ăn tập thể"],
    },
    "parking": {
        "train": ["bãi xe", "bãi đỗ xe", "nhà xe", "hầm xe", "chỗ để xe", "khu gửi xe"],
        "new": ["bãi giữ xe", "nhà để xe", "tầng hầm gửi xe", "chỗ gửi xe", "điểm gửi xe", "khu để xe máy",
                "nhà gửi xe", "bãi gửi xe", "hầm để xe", "lán xe", "bãi để xe", "khu để xe", "hầm gửi xe", "điểm trông xe", "bãi xe máy", "nhà xe sinh viên", "khu giữ xe", "bãi xe đạp"],
        "val": ["khu đậu xe", "nơi giữ xe"],
    },
    "lecture": {
        "train": ["giảng đường", "tòa giảng đường", "khu giảng đường", "hội trường học", "phòng học lớn", "lớp học"],
        "new": ["phòng học", "khu lớp học", "nhà học", "tòa nhà học", "hội trường", "giảng đường lớn",
                "giảng đường chính", "tòa học tập", "phòng học chung", "nhà giảng đường", "phòng học chính", "tòa lớp học", "phòng lý thuyết", "khu học tập", "giảng đường A", "phòng học đa năng", "khu nhà học", "lớp"],
        "val": ["dãy phòng học", "nơi lên lớp"],
    },
    "lab": {
        "train": ["phòng thí nghiệm", "phòng lab", "lab hóa", "khu thí nghiệm", "phòng thực hành", "xưởng thực hành"],
        "new": ["phòng nghiên cứu", "trung tâm thí nghiệm", "phòng máy", "xưởng", "phòng thí nghiệm hóa sinh",
                "phòng lab vật lý", "lab", "phòng thí nghiệm sinh học", "khu nghiên cứu", "phòng TN", "phòng thí nghiệm trung tâm", "phòng thực tập", "xưởng cơ khí", "lab điện tử", "phòng nghiên cứu khoa học", "xưởng thực tập", "phòng lab hóa", "lab sinh học"],
        "val": ["phòng thực nghiệm", "nơi làm thí nghiệm"],
    },
    "office": {
        "train": ["phòng hành chính", "khu hành chính", "tòa hiệu bộ", "phòng một cửa", "văn phòng khoa", "phòng đào tạo"],
        "new": ["văn phòng trường", "nhà điều hành", "phòng công tác sinh viên", "ban giám hiệu", "văn phòng",
                "bộ phận hành chính", "phòng tổ chức", "nhà hiệu bộ", "phòng kế toán", "văn phòng hành chính", "phòng tổ chức cán bộ", "phòng tài vụ", "văn phòng đoàn", "phòng công tác học sinh", "trụ sở hành chính", "khối văn phòng", "phòng quản trị", "phòng quản lý sinh viên"],
        "val": ["phòng giáo vụ", "nơi nộp hồ sơ"],
    },
    "gate": {
        "train": ["cổng trường", "cổng chính", "cổng vào", "cổng A", "lối vào trường", "chốt bảo vệ cổng"],
        "new": ["cổng sau", "cổng B", "cổng ra vào", "phòng bảo vệ", "cổng chào", "lối ra vào chính", "cổng phụ",
                "trạm bảo vệ", "cổng số 1", "lối vào chính", "cổng ngoài", "cổng lớn", "cửa ngõ trường", "lối vào", "cổng bảo vệ", "trạm gác cổng", "barie cổng", "cổng trước"],
        "val": ["nơi khách đến trường", "chốt cổng"],
    },
}

# Vai người nhận: cụm "{ALIAS}" được thay bằng một tên gọi của cùng loại.
ROLES = {
    "library": {"train": ["thủ thư", "cán bộ {ALIAS}"],
                "new": ["nhân viên thư viện", "cô thủ thư", "quản lý thư viện", "người trực quầy mượn sách", "người giữ sách", "cán bộ quản lý sách"], "val": []},
    "dorm": {"train": ["sinh viên nội trú", "ban quản lý {ALIAS}"],
             "new": ["bạn sinh viên ở KTX", "quản lý ký túc xá", "bác trực ký túc", "tổ tự quản KTX", "bạn cùng phòng ở KTX", "quản lý khu nội trú"],
             "val": ["trưởng phòng nội trú"]},
    "sports": {"train": ["đội bóng", "huấn luyện viên"],
               "new": ["thầy thể dục", "giáo viên thể chất", "câu lạc bộ bóng rổ", "vận động viên", "đội bóng chuyền", "đội tuyển điền kinh", "câu lạc bộ võ"],
               "val": []},
    "clinic": {"train": ["bác sĩ trường", "nhân viên y tế"],
               "new": ["điều dưỡng", "bác sĩ trực", "y sĩ", "nhân viên phòng khám", "y bác sĩ", "nhân viên trạm xá"], "val": ["y tá trực"]},
    "canteen": {"train": ["bếp trưởng", "cô cấp dưỡng"],
                "new": ["đầu bếp", "nhân viên nhà ăn", "cô nấu bếp", "chị bán cơm", "cô bán căng tin", "người phụ bếp"], "val": ["nhân viên quầy ăn"]},
    "parking": {"train": ["bác giữ xe", "nhân viên trông xe"],
                "new": ["người trông xe", "anh giữ xe", "chú bảo vệ bãi xe", "nhân viên bãi xe", "bác bảo vệ nhà xe", "chú trông xe"], "val": []},
    "lecture": {"train": ["giảng viên đang dạy", "lớp trưởng"],
                "new": ["thầy giáo đang giảng bài", "cô giáo trên lớp", "bí thư lớp", "sinh viên đang học", "giáo viên đứng lớp", "sinh viên lớp tôi"],
                "val": ["cán sự lớp"]},
    "lab": {"train": ["nhóm nghiên cứu", "kỹ thuật viên {ALIAS}"],
            "new": ["nghiên cứu viên", "trợ giảng thí nghiệm", "thầy phụ trách lab", "nhóm làm đề tài", "kỹ sư phòng lab", "sinh viên làm thí nghiệm"],
            "val": ["người hướng dẫn thực hành"]},
    "office": {"train": ["thư ký khoa", "cán bộ phòng đào tạo"],
               "new": ["cô văn thư", "nhân viên hành chính", "chánh văn phòng", "thầy trưởng phòng", "kế toán trường", "cán bộ văn phòng"],
               "val": ["chuyên viên một cửa"]},
    "gate": {"train": ["tổ đón khách", "bảo vệ cổng"],
             "new": ["chú bảo vệ", "anh bảo vệ trực", "nhân viên lễ tân", "đội an ninh", "chú gác cổng", "đội bảo vệ"], "val": ["nhân viên trực cổng"]},
}

ITEMS = {
    "library": ["chồng giáo trình", "sách tham khảo", "hộp sách trả", "thẻ thư viện", "luận văn", "tạp chí khoa học"],
    "dorm": ["chăn gối", "túi đồ giặt", "gói hàng đặt online", "bưu kiện", "quạt điện", "vali"],
    "sports": ["quả bóng", "bộ vợt", "dụng cụ tập luyện", "thùng nước uống", "áo thi đấu", "cờ hiệu"],
    "clinic": ["bộ băng gạc", "hộp thuốc", "túi sơ cứu", "hộp khẩu trang", "nhiệt kế", "hồ sơ khám bệnh"],
    "canteen": ["khay cơm", "thùng thực phẩm", "nguyên liệu nấu ăn", "két nước ngọt", "thùng rau", "bao gạo"],
    "parking": ["mũ bảo hiểm", "chìa khóa xe", "thẻ giữ xe", "bơm xe", "áo mưa", "dây sạc xe điện"],
    "lecture": ["máy chiếu", "hộp phấn", "micro không dây", "tập bài kiểm tra", "bút trình chiếu", "đề thi"],
    "lab": ["hóa chất", "mẫu vật", "bộ dụng cụ thí nghiệm", "linh kiện điện tử", "ống nghiệm", "kính hiển vi"],
    "office": ["con dấu", "hồ sơ", "phong bì công văn", "tập giấy tờ", "giấy chứng nhận", "bảng điểm"],
    "gate": ["sổ ghi khách", "bộ đàm", "biển tên khách", "thẻ khách", "đèn pin", "danh sách khách mời"],
    "generic": ["bưu kiện", "gói hàng", "thùng hàng", "hộp giấy", "túi đồ", "tập tài liệu", "đơn hàng cần trả",
                "hộp quà", "thư", "kiện hàng", "món đồ", "hộp nhỏ", "túi xách", "laptop"],
}

VERBS = {"train": ["giao", "gửi", "mang", "chuyển", "đưa"],
         "new": ["đem", "chở", "chuyển phát", "ship", "mang giúp"], "val": ["vận chuyển"]}
TO = {"train": ["tới", "đến"], "new": ["sang", "về", "cho"], "val": []}
PLEASE = {"train": [" giúp mình", ""], "new": [" giùm mình", " giúp tôi", " nhé", " giúp nha", " hộ mình"], "val": []}

# Khung câu nơi giao. {G}: nơi giao (kèm tham chiếu), {I}: món hàng, {V}: động từ, {TO}: giới từ, {ROLE}, {PL}.
GOAL_FRAMES = {
    "train": [
        "Điểm giao: {G}. Hàng: {I}.", "Điểm giao: {G}.", "Cần {I} ở {G}.", "Có đơn giao {I} ở {G}.",
        "{ROLE} ở {G} đang chờ {I}.", "{ROLE} cần nhận {I}.", "{V} {I} {TO} {G}{PL}.", "Nhờ robot {v} {I} qua {G}.",
    ],
    "new": [
        "Địa chỉ nhận: {G}.", "Nơi nhận hàng là {G}.", "Làm ơn {v} {I} {TO} {G}.", "{I} này cần tới {G}.",
        "Bạn {v} {I} {TO} {G} nhé.", "Đơn hàng {I} giao tại {G}.", "Người nhận ở {G}.", "Giao tận nơi tại {G}: {I}.",
        "{ROLE} đang đợi {I} ở {G}.", "Mình cần gửi {I} cho {ROLE} ở {G}.", "{G} đang đợi {I}.",
        "Nhờ robot {v} {I} {TO} {G}{PL}.", "Giao giúp {I} cho {ROLE}.", "{ROLE} nhờ mang {I} tới.",
        "Cần chuyển {I} cho {ROLE}.", "Nơi giao: {G}.", "Giao hàng tới {G}.", "Đơn này giao ở {G}, hàng là {I}.",
        "Hãy {v} {I} {TO} {G}.", "Nhận hàng tại {G}.", "{ROLE} ở {G} cần {I}.",
        "Bên nhận: {ROLE} ở {G}.", "Giao cho người ở {G}.", "{G} là nơi cần giao.", "Mục tiêu: {G}.",
        "Gửi về {G} giúp nhé.", "Phải giao {I} {TO} {G}.", "Đơn {I}, nơi nhận {G}.", "Cần mang {I} qua {G}.",
        "{ROLE} đặt {I}, giao ở {G}.", "Đơn của {ROLE}: {I}.",
    ],
    "val": [
        "Hàng cho {G}: {I}.", "Đích đến là {G}, hàng cần giao là {I}.", "{I} phải được đưa tới {G}.",
        "{I} phải được chuyển tới {G}.",
    ],
}
# ROLE-frames cho biết nơi giao chỉ qua người nhận (không cần {G}).
ROLE_ONLY = {"{ROLE} cần nhận {I}.", "Giao giúp {I} cho {ROLE}.", "{ROLE} nhờ mang {I} tới.", "Cần chuyển {I} cho {ROLE}.",
             "Đơn của {ROLE}: {I}."}

# Khung câu có cả điểm ghé {B} và nơi giao {G}.
VIA_GOAL_FRAMES = {
    "train": [
        "Trước khi {v} {I} {TO} {G}{PL}, nhớ ghé {B} lấy {J}.",
        "Ghé {B} lấy {J} trước, rồi {v} {I} {TO} {G}{PL}.",
        "Trước tiên qua {B} nhận {J} sau đó {v} {I} {TO} {G}{PL}.",
        "Trước tiên qua {B} nhận {J}, sau đó {v} {I} {TO} {G}{PL}.",
    ],
    "new": [
        "Đầu tiên ghé {B} lấy {J}, xong rồi mới {v} {I} {TO} {G}.",
        "{V} {I} {TO} {G}, nhưng nhớ ghé {B} lấy {J} trước.",
        "Ghé qua {B} trước đã rồi mới tới {G}.",
        "Sau khi lấy {J} ở {B}, hãy {v} {I} {TO} {G}.",
        "Phải qua {B} lấy {J} rồi mới {v} tới {G}.",
        "Trên đường tới {G}, ghé {B} lấy {J} trước nhé.",
        "Tới {B} trước, sau đó mới tới {G}.",
        "Trước hết dừng ở {B} lấy {J}, rồi {v} {I} {TO} {G}.",
        "Lộ trình: {B} trước, {G} sau.",
        "Đi {B} lấy {J} trước rồi qua {G}.",
        "Giao {I} {TO} {G} sau khi đã ghé {B}.",
        "Đến {B} trước rồi tới {G}.", "Ghé ngang {B} lấy {J} rồi hẵng đi {G}.",
        "Lấy {J} từ {B}, sau đó {v} {I} {TO} {G}.", "{V} {I} {TO} {G}; tuy nhiên phải ghé {B} trước.",
        "Trước khi đến {G}, ghé {B} một chút.", "Ghé {B} rồi mới tới {G} nhé.",
    ],
    "val": [
        "Chưa đi thẳng được: ghé {B} trước, rồi {v} {I} {TO} {G}.",
        "Lấy {J} ở {B} xong thì {v} {I} {TO} {G}.",
    ],
}
# Câu điểm ghé đứng riêng (đi kèm một câu nơi giao).
VIA_FRAMES = {
    "train": ["Nhớ ghé {B} lấy {J} trước.", "Nhưng phải tạt qua {B} trước đã."],
    "new": ["Trên đường đi nhớ rẽ vào {B} lấy {J} trước.", "Nhớ dừng ở {B} trước để lấy {J}.",
            "Cần lấy {J} tại {B} trước khi giao.", "Trước đó phải ghé {B} đã.", "Ghé {B} trước nhé.",
            "Nhưng trước tiên hãy qua {B} lấy {J}.", "Đừng quên ghé {B} trước.", "Trước khi giao, tạt qua {B} lấy {J}.",
            "Nhớ ghé ngang {B} trước.", "Trước tiên phải qua {B}.", "Đi qua {B} trước để lấy {J}.",
            "Có việc ở {B} nên ghé đó trước."],
    "val": [],
}
DISTRACTOR_FRAMES = {
    "train": ["Không cần ghé {X}.", "Hôm qua đã giao ở {X} rồi.", "Đừng nhầm với {X} nhé.", "Người nhận đã rời {X} rồi."],
    "new": ["Không phải giao ở {X} đâu.", "Khỏi ghé {X}.", "Đừng ghé {X}.", "{X} đã nhận hàng hôm trước rồi.",
            "Lần trước giao nhầm sang {X}, lần này đừng nhầm.", "{X} hôm nay đóng cửa.", "Không cần qua {X} nữa.",
            "Đơn cũ ở {X} đã xong rồi.", "Tránh nhầm sang {X}.", "Đơn này không liên quan tới {X}.",
            "Không giao cho {X}.", "Đừng đi {X}.", "Chỗ {X} không cần ghé đâu.", "Không cần đến {X}.",
            "Chú ý không phải {X}.", "{X} thì không liên quan.", "Người nhận không còn ở {X}.", "Đừng mang tới {X}.",
            "Hàng không phải cho {X}.", "Lưu ý: không ghé {X}."],
    "val": ["Bỏ qua {X}, không phải ở đó.", "{X} không phải điểm nhận."],
}

# Tham chiếu không gian, gắn ngay sau tên địa điểm. {A}: địa điểm mốc.
REFS = {
    "north": {"train": ["phía bắc", "nằm phía bắc", "ở phía trên bản đồ"],
              "new": ["ở phía bắc", "về phía bắc", "ở hướng bắc", "phía Bắc", "ở nửa trên bản đồ", "ở rìa trên bản đồ",
                      "phía trên", "ở phía trên", "ở góc trên", "phía trên cùng", "ở hàng trên", "tít phía bắc",
                      "ở cực bắc", "ở phần trên bản đồ"],
              "val": ["ở mạn trên", "mạn trên bản đồ"]},
    "south": {"train": ["phía nam", "nằm phía nam", "ở phía dưới bản đồ"],
              "new": ["ở phía nam", "về phía nam", "ở hướng nam", "phía Nam", "ở nửa dưới bản đồ", "ở rìa dưới bản đồ",
                      "phía dưới", "ở phía dưới", "ở góc dưới", "phía dưới cùng", "ở hàng dưới", "tít phía nam",
                      "ở cực nam", "ở phần dưới bản đồ"],
              "val": ["ở mạn dưới", "mạn dưới bản đồ"]},
    "west": {"train": ["phía tây", "nằm phía tây", "ở bên trái bản đồ"],
             "new": ["ở phía tây", "về phía tây", "ở hướng tây", "phía Tây", "ở nửa trái bản đồ", "ở rìa trái bản đồ",
                     "bên trái", "ở bên trái", "ở góc trái", "phía bên trái", "tít phía tây", "ở cực tây",
                     "ở phần bên trái bản đồ", "bên tay trái bản đồ"],
             "val": ["ở mạn trái", "mạn trái bản đồ"]},
    "east": {"train": ["phía đông", "nằm phía đông", "ở bên phải bản đồ"],
             "new": ["ở phía đông", "về phía đông", "ở hướng đông", "phía Đông", "ở nửa phải bản đồ", "ở rìa phải bản đồ",
                     "bên phải", "ở bên phải", "ở góc phải", "phía bên phải", "tít phía đông", "ở cực đông",
                     "ở phần bên phải bản đồ", "bên tay phải bản đồ"],
             "val": ["ở mạn phải", "mạn phải bản đồ"]},
    "near": {"train": ["gần {A} hơn", "nằm gần {A}", "ở gần {A}"],
             "new": ["gần với {A}", "cạnh {A}", "kế bên {A}", "ngay cạnh {A}", "gần {A} nhất", "nằm cạnh {A}",
                     "không xa {A}", "ở gần {A} hơn", "bên cạnh {A}", "liền kề {A}", "ở khu vực gần {A}",
                     "cách {A} không xa", "gần chỗ {A}", "cận kề {A}", "ở ngay gần {A}"],
             "val": ["sát {A}", "nằm sát {A}"]},
    "far": {"train": ["xa {A} hơn", "nằm xa {A}", "ở xa {A}"],
            "new": ["xa {A} nhất", "ở đầu kia so với {A}", "cách {A} xa hơn", "nằm xa {A} hơn", "ở xa {A} hơn",
                    "không gần {A}", "phía xa {A}", "xa hẳn {A}", "ở phía đối diện {A}", "tách xa {A}",
                    "cách {A} một đoạn dài", "xa tít {A}", "xa chỗ {A}", "ở xa khỏi {A}"],
            "val": ["cách xa {A}", "nằm cách xa {A}"]},
}

URGENT = {
    True: {"train": ["Hỏa tốc!", "Việc này khẩn cấp.", "Gấp nhé!", "Cần ngay trong 5 phút.", "Càng nhanh càng tốt.",
                     "Đang cần gấp lắm."],
           "new": ["Khẩn!", "Cần gấp.", "Giao ngay giúp mình.", "Đi nhanh lên nhé.", "Đơn hỏa tốc.",
                   "Trong 10 phút phải tới.", "Gấp lắm rồi!", "Ưu tiên giao trước.", "Người nhận đang đợi gấp.",
                   "Phải tới ngay lập tức.", "Đừng để trễ nhé.", "Không thể chờ được nữa.", "Chậm là không kịp đâu.",
                   "Không được để muộn.", "Rất gấp!", "Cần ngay bây giờ.", "Khẩn cấp lắm.", "Nhanh lên giúp mình.",
                   "Cần giao trong vòng 15 phút.", "Người nhận cần ngay.", "Đi ngay kẻo trễ.", "Đơn khẩn, làm liền nhé.",
                   "Gấp gấp!", "Phải có trong sáng nay.", "Không chờ được đâu.", "Không thể để lâu.",
                   "Đơn này rất cần nhanh.", "Tranh thủ giao liền nhé.", "Xử lý gấp giúp.", "Cần tới sớm nhất có thể.",
                   "Đơn cấp tốc.", "Hạn chót là 10 phút nữa.", "Đừng chần chừ.", "Không được để chờ lâu.",
                   "Làm ngay nhé.", "Đi liền giúp mình.",
                   "Không thể đợi tới chiều.", "Không còn thời gian đâu.", "Trễ là hỏng việc.", "Không kịp nữa rồi, đi ngay."],
           "val": ["Ưu tiên số một, đi ngay.", "Không được chậm trễ."]},
    False: {"train": ["Cứ từ từ, không vội.", "Không gấp đâu.", "Chiều nay giao cũng được."],
            "new": ["Không vội đâu.", "Thong thả nhé.", "Lúc nào rảnh thì giao.", "Không cần nhanh.",
                    "Tuần này giao là được.", "Không phải đơn gấp.", "Chậm một chút cũng không sao.", "Từ từ cũng được.",
                    "Không khẩn cấp.", "Không có gì gấp.", "Cuối ngày giao cũng được.", "Không cần gấp đâu.",
                    "Tối nay giao cũng ổn.", "Đơn này không gấp.", "Không có hạn gấp.", "Giao lúc nào cũng được.",
                    "Thoải mái thời gian.", "Không cần giao ngay.", "Hôm nào tiện thì giao.", "Đơn thường, không ưu tiên.",
                    "Không cần khẩn trương.", "Từ từ thôi.", "Chưa cần ngay đâu.", "Không phải hỏa tốc.",
                    "Sáng mai giao vẫn được.", "Chậm cũng được.", "Không gấp gì cả.", "Thư thả nhé.",
                    "Để chiều giao cũng không trễ.", "Chậm chút vẫn kịp.", "Không cần ngay đâu.", "Trễ một chút cũng được.",
                    "Tuần sau giao vẫn kịp."],
            "val": ["Mai giao cũng kịp.", "Không cần vội."]},
}
FRAGILE = {
    True: {"train": ["Chú ý: hàng dễ hỏng khi va đập.", "Bên trong là đồ thủy tinh.", "Hàng dễ vỡ, đi cẩn thận.",
                     "Đồ gốm bên trong, tránh va chạm.", "Nhẹ tay vì đồ rất dễ vỡ."],
           "new": ["Cẩn thận, hàng dễ vỡ.", "Bên trong có đồ sứ.", "Hàng thủy tinh, nâng niu giúp mình.",
                   "Đừng để rung lắc mạnh.", "Có chai lọ thủy tinh bên trong.", "Hàng dễ bể.", "Tránh xóc nảy nhé.",
                   "Đồ điện tử nhạy cảm, đi nhẹ nhàng.", "Không được làm rơi.", "Đồ dễ hư hỏng, cẩn thận.",
                   "Hàng dễ vỡ lắm.", "Đi chậm kẻo vỡ đồ.", "Đồ dễ bể, nhẹ tay.", "Bên trong là bình hoa.",
                   "Hàng dễ nứt vỡ.", "Đựng ly thủy tinh, cẩn thận.", "Cầm nhẹ nhàng giúp.",
                   "Có màn hình bên trong, tránh va đập.", "Đồ sứ, đi chậm thôi.", "Tránh để va chạm mạnh.",
                   "Hàng nhạy cảm với va đập.", "Đồ rất dễ hỏng.", "Đừng để rơi nhé.", "Hàng cần nâng niu.",
                   "Bên trong có chai thủy tinh, dễ vỡ.", "Cẩn thận kẻo bể.",
                   "Không chịu được va đập.", "Không được để va chạm.", "Rơi là hỏng ngay."],
           "val": ["Hàng kỵ va đập.", "Kiện hàng này rất mỏng manh."]},
    False: {"train": ["Đồ không vỡ được đâu.", "Hàng chắc chắn, không dễ vỡ."],
            "new": ["Hàng không dễ vỡ.", "Đồ cứng, không sợ vỡ.", "Không phải hàng dễ vỡ.", "Va chạm chút cũng không sao.",
                    "Hàng đóng gói kỹ, không lo hỏng.", "Đồ nhựa, không vỡ được.", "Không cần nâng niu.",
                    "Hàng bình thường, không dễ vỡ.", "Đồ chắc lắm, yên tâm.", "Hàng cứng cáp, không lo.",
                    "Không có gì dễ vỡ.", "Đồ vải, không vỡ được.", "Có rơi cũng không hỏng.", "Không cần cẩn thận quá.",
                    "Đồ chắc chắn lắm.", "Hàng không sợ va đập.", "Không phải đồ dễ bể.", "Va đập nhẹ cũng không hỏng.",
                    "Đồ giấy, không vỡ đâu.", "Yên tâm, hàng không dễ hỏng.", "Hàng an toàn, không dễ vỡ.",
                    "Có rơi cũng không vỡ.", "Không lo va đập.", "Không cần nhẹ tay đâu."],
            "val": ["Rơi cũng chẳng sao.", "Hàng bền, không lo vỡ."]},
}
GREETINGS = {"train": ["Xin chào!", "Robot ơi,", "Nhờ bạn nhé:", "Yêu cầu mới:"],
             "new": ["Chào bạn!", "Hey robot,", "Alo robot,", "Bạn robot ơi,", "Thông báo:"],
             "val": ["Chào robot!", "Nhắn robot:"]}
CLOSINGS = {"train": ["Cảm ơn!", "Cảm ơn nhiều.", "Thanks robot."],
            "new": ["Cám ơn bạn.", "Thank you!", "Cảm ơn robot nhé.", "Đa tạ!", "Cảm ơn bạn nhiều."], "val": []}


def pool(entry, use_val):
    """Ghép các nhóm của một mục từ điển."""
    out = list(entry["train"]) + list(entry["new"])
    if use_val:
        out += entry["val"]
    return out


# ----------------------------------------------------------------------------- tên gọi theo chức năng
# "nơi/chỗ/phòng/khu + hoạt động": sinh tên gọi mới thay vì chỉ học thuộc danh sách.
FUNC_PREFIX = ["nơi", "chỗ", "phòng", "khu", "khu vực", "điểm"]
FUNC_ACTS = {
    "library": {"new": ["mượn sách", "trả sách", "đọc sách", "mượn giáo trình", "tra cứu tài liệu", "mượn tài liệu",
                        "tự học", "đọc báo", "mượn truyện"], "val": ["mượn sách tham khảo"]},
    "dorm": {"new": ["sinh viên ở", "ngủ nghỉ của sinh viên", "lưu trú", "sinh viên nội trú ở", "nghỉ ngơi buổi tối"],
             "val": ["ở của sinh viên"]},
    "sports": {"new": ["tập thể dục", "chơi bóng", "thi đấu", "tập luyện thể thao", "đá bóng", "chơi cầu lông"],
               "val": ["vận động", "luyện tập"]},
    "clinic": {"new": ["khám bệnh", "sơ cứu", "lấy thuốc", "chữa bệnh", "kiểm tra sức khỏe", "băng bó vết thương",
                       "tiêm phòng"], "val": ["khám sức khỏe"]},
    "canteen": {"new": ["ăn cơm", "ăn uống", "mua đồ ăn", "nấu ăn", "ăn sáng", "bán cơm"],
                "val": ["phục vụ bữa trưa", "ăn trưa"]},
    "parking": {"new": ["gửi xe", "để xe", "đỗ xe", "trông xe", "cất xe", "sạc xe điện"], "val": ["giữ xe", "đậu xe"]},
    "lecture": {"new": ["học lý thuyết", "giảng bài", "thi cuối kỳ", "nghe giảng", "học chuyên đề", "học tiết lớn"],
                "val": ["lên lớp"]},
    "lab": {"new": ["thực hành", "nghiên cứu", "làm thực hành hóa", "đo đạc mẫu", "chạy thí nghiệm"],
            "val": ["làm thí nghiệm"]},
    "office": {"new": ["làm thủ tục", "xin giấy xác nhận", "nộp giấy tờ", "đóng học phí", "đóng dấu",
                       "xin bảng điểm"], "val": ["nộp hồ sơ"]},
    "gate": {"new": ["ra vào trường", "đón khách", "kiểm soát ra vào", "đăng ký khách", "khách làm thủ tục vào"],
             "val": ["khách đến trường"]},
}

# ----------------------------------------------------------------------------- ngữ pháp gấp / dễ vỡ
# Mỗi từ khóa xuất hiện ở cả hai phía để model không học tắt theo một từ. "val" = từ chỉ thấy ở validation.
ATTR_WORDS = {
    "fast_adj": {"new": ["gấp", "khẩn", "khẩn cấp", "hỏa tốc", "cấp tốc", "gấp rút"], "val": []},
    "fast_adv": {"new": ["ngay", "liền", "ngay lập tức", "sớm nhất có thể", "càng sớm càng tốt"], "val": []},
    "slow_word": {"new": ["chậm", "trễ", "muộn", "để lâu", "chần chừ"], "val": ["chậm trễ"]},
    "time": {"new": ["Chiều nay", "Tối nay", "Tuần sau", "Cuối tuần", "Ngày mốt", "Lúc nào", "Hôm khác", "Cuối ngày"],
             "val": ["Mai", "Ngày mai"]},
    "ok_end": {"new": ["cũng được", "vẫn được", "cũng ổn", "vẫn ổn", "cũng không sao"], "val": ["cũng kịp", "vẫn kịp"]},
    "bad": {"new": ["hỏng việc", "rắc rối to", "bị phạt", "người nhận phàn nàn", "lỡ việc"], "val": []},
    "dmg": {"new": ["vỡ", "bể", "hỏng", "nứt", "móp", "trầy"], "val": []},
    "impact": {"new": ["va đập", "va chạm", "rung lắc", "xóc", "rơi"], "val": []},
    "material": {"new": ["đồ thủy tinh", "đồ sứ", "bình gốm", "chai lọ", "màn hình máy tính", "trứng", "ly tách",
                         "gương soi"], "val": []},
    "care": {"new": ["Nhẹ tay", "Cẩn thận", "Nâng niu", "Đi chậm", "Đi êm"], "val": []},
    "robust": {"new": ["chắc chắn", "cứng cáp", "đóng gói rất kỹ", "bằng nhựa", "bằng vải", "bằng giấy", "bằng sắt"],
               "val": ["bền"]},
    "intens": {"new": ["", "rất ", "khá ", "cực kỳ ", "hơi "], "val": []},
}
URGENT_GRAMMAR = {
    True: ["{fast_adj_cap}!", "Đơn này {intens}{fast_adj}.", "Không được {slow_word}.", "Đừng để {slow_word}.",
           "Không thể {slow_word} được.", "{slow_word_cap} là {bad}.", "Cần giao {fast_adv}.", "Đi {fast_adv} nhé.",
           "Phải tới trong {n} phút.", "Người nhận cần {fast_adv}.", "Việc này {intens}{fast_adj}, đi {fast_adv}.",
           "Không còn thời gian, đi {fast_adv}."],
    False: ["Không {fast_adj} đâu.", "Không {fast_adj} lắm.", "Không cần {fast_adj}.", "Không cần giao {fast_adv}.",
            "{time} giao {ok_end}.", "{slow_word_cap} một chút {ok_end}.", "Cứ từ từ, {slow_word} chút {ok_end}.",
            "Đơn này không {fast_adj}, {time_low} giao {ok_end}.", "Không có gì {fast_adj}.",
            "Chưa cần {fast_adv} đâu."],
}
FRAGILE_GRAMMAR = {
    True: ["Hàng {intens}dễ {dmg}.", "Đừng để {dmg}.", "Không được làm {dmg}.", "Tránh {impact} nhé.",
           "Bên trong là {material}.", "{impact_cap} là {dmg} ngay.", "Hàng sợ {impact}.",
           "Hàng không chịu được {impact}.", "{care} giúp mình.", "{care}, hàng dễ {dmg}.",
           "Đồ dễ {dmg}, đừng để {impact}."],
    False: ["Hàng không dễ {dmg}.", "Hàng không {dmg} được đâu.", "Hàng {robust}.", "Đồ {robust}, không lo {dmg}.",
            "{impact_cap} {ok_end}.", "Hàng chịu {impact} tốt.", "Không sợ {impact}.",
            "Không cần {care_low}.", "Có {impact} chút cũng không {dmg}.", "Hàng {robust}, không dễ {dmg}."],
}
SPATIAL_EXTRA = {
    "north": ["ở mé bắc", "vùng phía bắc", "ở đầu phía bắc", "khu phía bắc", "ở phần phía bắc"],
    "south": ["ở mé nam", "vùng phía nam", "ở đầu phía nam", "khu phía nam", "ở phần phía nam"],
    "west": ["ở mé tây", "vùng phía tây", "ở đầu phía tây", "khu phía tây", "ở phần phía tây"],
    "east": ["ở mé đông", "vùng phía đông", "ở đầu phía đông", "khu phía đông", "ở phần phía đông"],
    "near": ["cách {A} vài bước", "chỉ cách {A} một đoạn ngắn", "ở liền bên {A}", "ngay gần {A}"],
    "far": ["cách {A} cả một quãng", "xa tận đầu kia so với {A}", "xa khỏi {A}"],
}
