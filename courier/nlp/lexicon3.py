"""Từ điển bổ sung cho vòng v3 (private test): mô tả nơi giao qua bản đồ, đính chính, đối chiếu.

Nhóm:
  "train": cụm có trong train v3 (đã kiểm tra bằng scripts/nlp_analyze.py)
  "new":   cụm tự nghĩ ra (kiến thức tiếng Việt chung), KHÔNG lấy từ validation/test v3
  "val":   cụm chỉ thấy ở validation v3 -> chỉ dùng cho model cuối (train+val), bị loại ở chế độ holdout
Không có cụm nào lấy từ test.
"""

# ----------------------------------------------------------------------------- danh từ chung chỉ "một nơi"
# Dùng cho mô tả nơi giao qua bản đồ (không nêu loại địa điểm).
GENERIC_PLACE = {
    "train": ["địa điểm", "nơi", "chỗ"],
    "new": ["điểm", "vị trí", "khu", "khu vực", "tòa nhà", "công trình", "điểm đến", "địa chỉ", "nơi nằm",
            "địa điểm nào", "chỗ nào", "nơi nào", "căn nhà", "tòa", "điểm nhận", "chốn"],
    "val": [],
}

# ----------------------------------------------------------------------------- *_most (không kèm loại)
# {P}: danh từ chung. Train có đúng 3 cách cho mỗi hướng.
MOST = {
    "north_most": {
        "train": ["{P} ở trên cùng bản đồ", "{P} xa nhất về phía bắc", "{P} nằm cao nhất trên bản đồ"],
        "new": ["{P} nằm trên cùng bản đồ", "{P} nằm xa nhất về phía bắc", "{P} cao nhất trên bản đồ",
                "{P} nằm sát mép trên nhất", "{P} gần mép trên nhất", "{P} ở rìa trên cùng",
                "{P} ngoài cùng phía trên", "{P} ở cực bắc bản đồ", "{P} phía bắc nhất", "{P} ở phía bắc nhất",
                "{P} nằm ở hàng trên cùng", "{P} ở tít trên cùng", "{P} ở vị trí cao nhất trên bản đồ",
                "{P} xa nhất về hướng bắc", "{P} nằm xa nhất phía trên", "{P} ở trên cùng",
                "{P} nằm ở mép trên cùng của bản đồ", "{P} sát rìa trên bản đồ nhất", "{P} nằm cao nhất",
                "{P} ở đỉnh bản đồ", "{P} trên cùng của sơ đồ", "{P} nằm cao nhất trên sơ đồ",
                "{P} ở phần trên cùng bản đồ", "{P} nằm phía trên cùng", "{P} xa nhất lên phía trên",
                "{P} ở mạn bắc xa nhất", "{P} bắc nhất trên bản đồ", "{P} nằm gần cạnh trên bản đồ nhất",
                "{P} ở trên cao nhất bản đồ", "{P} có vị trí cao nhất"],
        "val": ["{P} ở rìa trên nhất", "{P} sát mép trên nhất"],
    },
    "south_most": {
        "train": ["{P} ở dưới cùng bản đồ", "{P} xa nhất về phía nam", "{P} nằm thấp nhất trên bản đồ"],
        "new": ["{P} nằm dưới cùng bản đồ", "{P} nằm xa nhất về phía nam", "{P} thấp nhất trên bản đồ",
                "{P} nằm sát mép dưới nhất", "{P} gần mép dưới nhất", "{P} ở rìa dưới cùng",
                "{P} ngoài cùng phía dưới", "{P} ở cực nam bản đồ", "{P} phía nam nhất", "{P} ở phía nam nhất",
                "{P} nằm ở hàng dưới cùng", "{P} ở tít dưới cùng", "{P} ở vị trí thấp nhất trên bản đồ",
                "{P} xa nhất về hướng nam", "{P} nằm xa nhất phía dưới", "{P} ở dưới cùng",
                "{P} nằm ở mép dưới cùng của bản đồ", "{P} sát rìa dưới bản đồ nhất", "{P} nằm thấp nhất",
                "{P} ở đáy bản đồ", "{P} dưới cùng của sơ đồ", "{P} nằm thấp nhất trên sơ đồ",
                "{P} ở phần dưới cùng bản đồ", "{P} nằm phía dưới cùng", "{P} xa nhất xuống phía dưới",
                "{P} ở mạn nam xa nhất", "{P} nam nhất trên bản đồ", "{P} nằm gần cạnh dưới bản đồ nhất",
                "{P} ở dưới thấp nhất bản đồ", "{P} có vị trí thấp nhất"],
        "val": ["{P} ở rìa dưới nhất", "{P} sát mép dưới nhất"],
    },
    "west_most": {
        "train": ["{P} xa nhất về phía tây", "{P} ngoài cùng bên trái", "{P} nằm sát mép trái nhất"],
        "new": ["{P} nằm xa nhất về phía tây", "{P} nằm ngoài cùng bên trái", "{P} ở ngoài cùng bên trái",
                "{P} gần mép trái nhất", "{P} ở rìa trái cùng", "{P} ở cực tây bản đồ", "{P} phía tây nhất",
                "{P} ở phía tây nhất", "{P} nằm ở cột ngoài cùng bên trái", "{P} ở tít bên trái",
                "{P} bên trái nhất", "{P} nằm bên trái nhất bản đồ", "{P} xa nhất về hướng tây",
                "{P} xa nhất về bên trái", "{P} ở bên trái cùng", "{P} nằm ở mép trái của bản đồ",
                "{P} sát rìa trái bản đồ nhất", "{P} ở trái cùng bản đồ", "{P} ngoài cùng phía tây",
                "{P} ở phần ngoài cùng bên trái", "{P} tây nhất trên bản đồ", "{P} nằm gần cạnh trái bản đồ nhất",
                "{P} ở mạn tây xa nhất", "{P} nằm lệch trái nhất", "{P} ở góc trái xa nhất",
                "{P} có vị trí xa nhất về bên trái", "{P} nằm phía trái cùng", "{P} sát lề trái nhất"],
        "val": ["{P} ở rìa trái nhất", "{P} sát mép trái nhất"],
    },
    "east_most": {
        "train": ["{P} xa nhất về phía đông", "{P} ngoài cùng bên phải", "{P} nằm sát mép phải nhất"],
        "new": ["{P} nằm xa nhất về phía đông", "{P} nằm ngoài cùng bên phải", "{P} ở ngoài cùng bên phải",
                "{P} gần mép phải nhất", "{P} ở rìa phải cùng", "{P} ở cực đông bản đồ", "{P} phía đông nhất",
                "{P} ở phía đông nhất", "{P} nằm ở cột ngoài cùng bên phải", "{P} ở tít bên phải",
                "{P} bên phải nhất", "{P} nằm bên phải nhất bản đồ", "{P} xa nhất về hướng đông",
                "{P} xa nhất về bên phải", "{P} ở bên phải cùng", "{P} nằm ở mép phải của bản đồ",
                "{P} sát rìa phải bản đồ nhất", "{P} ở phải cùng bản đồ", "{P} ngoài cùng phía đông",
                "{P} ở phần ngoài cùng bên phải", "{P} đông nhất trên bản đồ", "{P} nằm gần cạnh phải bản đồ nhất",
                "{P} ở mạn đông xa nhất", "{P} nằm lệch phải nhất", "{P} ở góc phải xa nhất",
                "{P} có vị trí xa nhất về bên phải", "{P} nằm phía phải cùng", "{P} sát lề phải nhất"],
        "val": ["{P} ở rìa phải nhất", "{P} sát mép phải nhất"],
    },
}

# ----------------------------------------------------------------------------- anchor_near (không kèm loại)
ANCHOR_NEAR = {
    "train": ["{P} cạnh {A} nhất", "{P} gần {A} nhất", "{P} nằm sát {A} nhất"],
    "new": ["{P} kề {A} nhất", "{P} sát {A} nhất", "{P} gần với {A} nhất", "{P} ở gần {A} nhất",
            "{P} nằm gần {A} nhất", "{P} nằm cạnh {A} nhất", "{P} ở sát {A} nhất", "{P} gần sát {A} nhất",
            "{P} cách {A} gần nhất", "{P} ở cạnh {A} nhất", "{P} nằm kề {A} nhất", "{P} sát bên {A} nhất",
            "{P} gần {A} nhất trên bản đồ", "{P} cận kề {A} nhất", "{P} liền kề {A} nhất", "{P} kế {A} nhất",
            "{P} gần chỗ {A} nhất", "{P} có khoảng cách tới {A} ngắn nhất", "{P} cách {A} ít nhất",
            "{P} nằm gần {A} hơn cả", "{P} gần {A} hơn tất cả các nơi khác", "{P} ở ngay sát {A}",
            "{P} láng giềng gần nhất của {A}", "{P} gần kề {A} nhất", "{P} ở bên cạnh {A} nhất",
            "{P} kế bên {A} nhất", "{P} sát vách {A} nhất", "{P} gần {A} nhất (không tính {A})",
            "{P} nằm gần {A} nhất trên sơ đồ", "{P} gần nhất với {A}", "{P} ở gần nhất so với {A}"],
    "val": [],
}

# ----------------------------------------------------------------------------- tham chiếu kèm loại (bổ sung)
# Phrase gắn sau tên địa điểm có hai bản (kind cũ). Thêm dạng so sánh nhất/so sánh hơn.
REFS_EXTRA = {
    "north": ["xa nhất về phía bắc", "nằm cao hơn", "ở phía trên hơn", "nằm trên", "ở mé trên bản đồ",
              "ở nửa bắc", "phía bắc hơn", "nằm phía bắc hơn", "ở vị trí cao hơn", "cao hơn trên bản đồ",
              "nằm cao nhất", "ở trên cùng", "ở hướng bắc bản đồ", "bên trên"],
    "south": ["xa nhất về phía nam", "nằm thấp hơn", "ở phía dưới hơn", "nằm dưới", "ở mé dưới bản đồ",
              "ở nửa nam", "phía nam hơn", "nằm phía nam hơn", "ở vị trí thấp hơn", "thấp hơn trên bản đồ",
              "nằm thấp nhất", "ở dưới cùng", "ở hướng nam bản đồ", "bên dưới"],
    "west": ["xa nhất về phía tây", "nằm bên trái hơn", "lệch trái hơn", "ở mé trái bản đồ", "ở nửa tây",
             "phía tây hơn", "nằm phía tây hơn", "ở bên trái cùng", "ngoài cùng bên trái", "ở hướng tây bản đồ",
             "phía tay trái"],
    "east": ["xa nhất về phía đông", "nằm bên phải hơn", "lệch phải hơn", "ở mé phải bản đồ", "ở nửa đông",
             "phía đông hơn", "nằm phía đông hơn", "ở bên phải cùng", "ngoài cùng bên phải", "ở hướng đông bản đồ",
             "phía tay phải"],
    "near": ["gần {A} hơn cả", "kề {A}", "nằm kề {A}", "gần sát {A}", "ở sát {A}", "sát bên {A}",
             "cách {A} gần hơn", "ở phía gần {A}", "gần phía {A}", "kế {A}", "gần {A}"],
    "far": ["xa {A}", "cách xa {A} hơn", "xa {A} hơn cả", "ở phía xa {A}", "nằm xa {A} hơn", "ở đầu xa {A}",
            "không phải cái gần {A}", "xa phía {A}", "cách {A} xa nhất", "nằm cách xa {A}"],
}

# ----------------------------------------------------------------------------- đính chính (bọc câu chính)
# {X}: nơi sai, {CORE}: phần yêu cầu đúng.
CORRECTION_PREFIX = {
    "train": ["Lúc nãy nhắn nhầm là {X}. Đúng ra: {CORE}", "Hủy đơn giao {X}. Thay vào đó: {CORE}"],
    "new": ["Tin nhắn trước ghi nhầm {X}. Sửa lại: {CORE}", "Xin lỗi, lúc nãy ghi sai thành {X}. Chính xác là: {CORE}",
            "Đính chính: không phải {X}. {CORE}", "Bỏ đơn giao {X} nhé. Đơn mới: {CORE}",
            "Không giao {X} nữa. Thay vào đó: {CORE}", "Đơn tới {X} bị hủy. Giờ thì: {CORE}",
            "Cập nhật: địa chỉ cũ {X} sai rồi. {CORE}", "Mình nhắn nhầm {X}, đúng phải là: {CORE}",
            "Nhầm chút: {X} là sai. {CORE}", "Đổi điểm giao, không còn là {X}. {CORE}",
            "Thôi khỏi giao {X}. Đổi sang: {CORE}", "Tin trước nói {X} là sai nhé. {CORE}",
            "Hủy yêu cầu cũ ở {X}. Yêu cầu mới: {CORE}", "Lúc nãy bảo giao {X} là nhầm. {CORE}",
            "{X} là địa chỉ cũ, bỏ đi. {CORE}", "Đơn trước gửi {X} hủy rồi. {CORE}",
            "Sửa lại đơn: không phải {X}. {CORE}", "Lúc trước ghi {X}, giờ sửa: {CORE}",
            "Ban nãy nói nhầm là {X}. Chính xác: {CORE}", "Hủy giao {X}. Thay bằng: {CORE}",
            "Điểm giao cũ ({X}) bị hủy. {CORE}", "Thông tin {X} lúc nãy sai. Đúng là: {CORE}",
            "Bỏ qua tin nhắn về {X}. Mới nhất: {CORE}", "Không phải {X} như đã nhắn. {CORE}",
            "Ghi nhầm {X} rồi, sửa: {CORE}", "Đơn giao {X} đã hủy. Bây giờ: {CORE}",
            "Lúc nãy nhắn nhầm {X}. Đúng là: {CORE}", "Hủy đơn đến {X}. Thay vào đó, {CORE}"],
    "val": ["Không giao {X} nữa nhé, đổi lại: {CORE}"],
}
CORRECTION_SUFFIX = {
    "train": ["(tin trước ghi {X} là nhầm)"],
    "new": ["({X} là ghi nhầm)", "(trước đó ghi nhầm {X})", "(địa chỉ {X} lúc trước là sai)", "(đã sửa từ {X})",
            "(không phải {X} như tin trước)", "(sửa lại: không phải {X})", "(tin cũ ghi {X} là sai)",
            "(lúc nãy ghi {X} là nhầm)", "(đơn cũ ở {X} đã hủy)", "(bỏ {X} đi nhé)", "(nhắn nhầm {X} lúc trước)",
            "(tin nhắn trước nhầm {X})", "({X} trong tin trước là nhầm)", "(trước có ghi {X}, bỏ qua nhé)",
            "Tin trước ghi {X} là nhầm.", "Lúc nãy ghi {X} là sai nhé.", "{X} trong tin trước là nhầm."],
    "val": [],
}

# ----------------------------------------------------------------------------- đối chiếu (thay cho {G})
CONTRAST = {
    "train": ["không phải {X} mà là {G}", "{G} chứ không phải {X}"],
    "new": ["{G} (không phải {X})", "{G}, không phải {X}", "{G} chứ không phải là {X}", "{G} thay vì {X}",
            "không phải {X}, mà là {G}", "{G} chứ không là {X}", "chính xác là {G} chứ không phải {X}",
            "đúng là {G}, không phải {X}", "{G} - không phải {X} đâu", "không phải là {X} mà là {G}",
            "{G}, chứ đâu phải {X}", "{G} chứ chẳng phải {X}", "không phải {X} đâu, mà là {G}",
            "{G} chứ không phải ở {X}", "không phải ở {X} mà ở {G}", "{G}, không phải ở {X}",
            "{G} (chứ không phải {X})", "đâu phải {X}, mà là {G}"],
    "val": ["{G}, không phải {X}"],
}

# ----------------------------------------------------------------------------- câu gây nhiễu (bổ sung)
DISTRACTOR_EXTRA = {
    "new": ["Không cần ghé {X} đâu.", "Đừng nhầm sang {X}.", "Người nhận không còn ở {X} nữa.",
            "Hôm trước đã giao tới {X} rồi.", "Không giao tới {X}.", "Đơn ở {X} đã giao xong.",
            "Không phải {X} đâu nhé.", "Đừng ghé {X} nhé.", "Không cần tạt qua {X}.",
            "Người nhận vừa rời {X}.", "{X} không cần giao.", "Tuần trước đã giao cho {X} rồi.",
            "Bỏ qua {X}.", "Khỏi qua {X} nhé.", "Đừng dừng ở {X}.", "Không ghé {X} đâu.",
            "Không dừng lại ở {X}.", "{X} không liên quan đơn này.", "Đừng đi nhầm vào {X}.",
            "Hàng không giao ở {X}.", "Người nhận đã chuyển khỏi {X}.", "Không còn ai ở {X} nhận đâu.",
            "Đơn ở {X} hôm qua xong rồi.", "Không cần đi ngang {X}.", "Không được giao nhầm sang {X}.",
            "Chú ý đừng nhầm với {X}.", "Lưu ý: đừng nhầm {X}.", "Không cần rẽ vào {X}.",
            "Không giao {X} nữa.", "Giao ở {X} là sai.", "Không ghé {X}, không phải ở đó.",
            "{X} đã nhận đủ hàng rồi.", "Không cần mang gì tới {X}.", "{X} đóng cửa rồi, khỏi ghé.",
            "Hôm qua {X} đã nhận rồi."],
    "val": [],
}

# ----------------------------------------------------------------------------- khung câu nơi giao (bổ sung)
GOAL_FRAMES_EXTRA = {
    "new": ["Giao {I} tới {G} nhé.", "{I} gửi tới {G}.", "Cần đưa {I} tới {G}.", "Mang {I} sang {G} giúp mình.",
            "Đơn hàng: {I}, giao tới {G}.", "Địa điểm nhận: {G}.", "Chỗ giao: {G}. Hàng: {I}.",
            "Nơi nhận: {G}. Món hàng: {I}.", "Giao hàng: {I} -> {G}.", "Điểm đến: {G}.",
            "Đích: {G}. Đồ: {I}.", "Đem {I} qua {G} nhé.", "Chở {I} tới {G}.", "{I} cần chuyển tới {G}.",
            "Có {I} cần giao tới {G}.", "Mình có {I} cần đưa đến {G}.", "Có người ở {G} chờ {I}.",
            "Người nhận đang ở {G}.", "Giao đến {G}, hàng là {I}.", "Nhận tại {G}.",
            "Nhờ robot mang {I} tới {G}.", "Robot mang {I} đến {G} nhé.", "Bạn chuyển giúp {I} tới {G}.",
            "Gửi {I} tới {G} hộ mình.", "Cần giao {I} tới {G}.", "{G} cần {I}.", "{G} đặt {I}.",
            "Đơn giao tới {G}.", "Có hàng cho {G}.", "Có một {I} giao tới {G}.", "Giao {I} cho người ở {G}.",
            "Hàng gửi đến {G}: {I}.", "Đi tới {G} giao {I}.", "Đến {G} giao {I} nhé.", "Tới {G} giao {I}.",
            "Địa chỉ giao hàng là {G}.", "Đơn {I} cần tới {G}.", "Mang tới {G}: {I}."],
    "val": ["Hàng cho {G}: {I}.", "{I} phải được {v} tới {G}.", "Đích đến là {G}, hàng cần giao là {I}."],
}
# khung câu có người nhận (lộ loại nơi giao), có {G} -> dùng được với mô tả qua bản đồ
ROLE_G_FRAMES = {
    "train": ["{ROLE} ở {G} đang chờ {I}."],
    "new": ["{ROLE} ở {G} đang đợi {I}.", "Giao {I} cho {ROLE} ở {G}.", "{ROLE} tại {G} cần {I}.",
            "Mang {I} tới {G} cho {ROLE}.", "{ROLE} đang ở {G}, cần {I}.", "Người nhận là {ROLE} ở {G}.",
            "Gửi {I} cho {ROLE} đang ở {G}.", "{ROLE} ngồi ở {G} chờ {I}.", "{ROLE} làm việc ở {G} cần nhận {I}.",
            "Chuyển {I} đến {G}, giao tận tay {ROLE}."],
    "val": [],
}

VIA_GOAL_FRAMES_EXTRA = {
    "new": ["Trước khi tới {G}, nhớ ghé {B} lấy {J}.", "Ghé {B} trước, sau đó mới {v} {I} {TO} {G}.",
            "{V} {I} {TO} {G}, nhưng trước đó ghé {B} lấy {J}.", "Lấy {J} ở {B} rồi mới {v} {I} {TO} {G}.",
            "Trước khi {v} {I} {TO} {G}, tạt qua {B} một lát.", "Phải ghé {B} trước khi tới {G}.",
            "Ghé {B} đã, rồi {v} {I} {TO} {G}.", "Đường đi: {B} rồi tới {G}.", "Qua {B} lấy {J}, xong {v} {I} {TO} {G}.",
            "Trước tiên ghé {B}, sau đó tới {G}.", "{V} {I} {TO} {G}. Trước đó nhớ tạt qua {B}.",
            "Hãy ghé {B} trước rồi mới tới {G}.", "Đi {G} nhưng ghé {B} trước nhé.",
            "Trên đường đến {G}, tạt vào {B} lấy {J}.", "Trước khi đến {G} giao {I}, qua {B} lấy {J} đã."],
    "val": ["Ghé {B} trước, rồi {v} {I} {TO} {G}{PL}.", "Ghé {B} lấy {J} trước, rồi {v} {I} {TO} {G}{PL}."],
}
VIA_FRAMES_EXTRA = {
    "new": ["Trước đó ghé {B} lấy {J} nhé.", "Nhớ tạt qua {B} trước.", "Phải ghé {B} lấy {J} trước khi giao.",
            "Ghé {B} trước để lấy {J}.", "Trên đường ghé {B} lấy {J}.", "Trước khi giao nhớ ghé {B}.",
            "Cần qua {B} trước đã.", "Nhớ dừng ở {B} trước nhé.", "Đi qua {B} trước rồi hẵng giao.",
            "Lấy {J} ở {B} trước nhé.", "Trước khi tới nơi, ghé {B} một chút."],
    "val": [],
}

# ----------------------------------------------------------------------------- tên gọi / vai bổ sung
ALIASES_EXTRA = {
    "library": ["thư viện chính", "phòng thư viện", "khu thư viện", "kho tài liệu", "nhà sách của trường"],
    "dorm": ["ký túc", "khu ở sinh viên", "nhà nội trú sinh viên", "dãy ký túc xá", "tòa nhà ký túc"],
    "sports": ["nhà thể dục", "khu thể dục thể thao", "nhà đa năng", "trung tâm thể thao", "sân thi đấu"],
    "clinic": ["phòng y tế trường", "trạm y tế sinh viên", "phòng khám sinh viên", "y tế"],
    "canteen": ["nhà ăn trường", "căn tin trường", "khu căn tin", "bếp ăn sinh viên", "quán cơm"],
    "parking": ["bãi đậu xe", "bãi xe sinh viên", "nhà gửi xe sinh viên", "hầm để xe máy", "điểm đỗ xe"],
    "lecture": ["tòa giảng", "nhà giảng đường", "khu phòng học", "hội trường lớn", "dãy giảng đường"],
    "lab": ["phòng thí nghiệm chung", "phòng lab chung", "khu lab", "phòng thực hành hóa", "xưởng thí nghiệm"],
    "office": ["văn phòng khoa", "phòng giáo vụ khoa", "phòng hành chính tổng hợp", "khu văn phòng", "phòng đào tạo trường"],
    "gate": ["cổng chính trường", "cổng ra vào trường", "chốt bảo vệ", "trạm gác", "cổng phía trước"],
}
ROLES_EXTRA = {
    "library": ["thủ thư trực quầy", "chị thủ thư", "bạn trực thư viện"],
    "dorm": ["bạn ở phòng nội trú", "quản lý nội trú", "sinh viên ở ký túc"],
    "sports": ["thầy huấn luyện", "đội bóng đá", "câu lạc bộ thể thao", "đội tuyển bóng rổ"],
    "clinic": ["bác sĩ trực", "cô y tá", "nhân viên trạm y tế"],
    "canteen": ["cô nấu ăn", "anh phụ bếp", "quản lý nhà ăn"],
    "parking": ["chú giữ xe", "bác trông xe", "người giữ xe"],
    "lecture": ["giảng viên trên lớp", "thầy đang dạy", "lớp phó học tập"],
    "lab": ["kỹ thuật viên phòng thí nghiệm", "trưởng nhóm nghiên cứu", "thầy hướng dẫn lab"],
    "office": ["thư ký văn phòng", "cán bộ hành chính", "chuyên viên đào tạo"],
    "gate": ["bảo vệ trực cổng", "tổ lễ tân", "chú bảo vệ cổng"],
}

# vai người nhận chỉ thấy ở validation v3: dùng để GÁN NHÃN (labels.py) và cho model cuối (--use-val)
ROLES_VAL3 = {"sports": ["câu lạc bộ cầu lông"]}

GREETINGS_EXTRA = {"new": ["Robot à,", "Ê robot,", "Gửi robot:", "Robot giúp nhé:", "Có việc cho bạn:", "Này robot,",
                           "Đơn mới:", "Nhờ chút:", "Chào robot nhé,"], "val": ["Nhắn robot:", "Chào robot,"]}
CLOSINGS_EXTRA = {"new": ["Cảm ơn robot!", "Thanks!", "Cám ơn nhiều nhé.", "Nhờ cả vào bạn.", "Thank you robot.",
                          "Cảm ơn trước nhé.", "Thanks nhiều."], "val": []}

# câu đệm vô hại (nhiễu)
FILLERS = ["Trời hôm nay đẹp quá.", "Mình đang bận họp.", "Có gì nhắn lại mình nhé.", "Gọi mình nếu cần.",
           "Mình sẽ thanh toán sau.", "Phí giao mình trả rồi.", "Nhớ chụp ảnh khi giao.", "Giao xong báo mình nhé.",
           "Mã đơn ghi trên hộp.", "Người nhận sẽ ký xác nhận.", "Có ghi chú dán trên hộp."]
