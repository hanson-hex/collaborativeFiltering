/*
 Navicat Premium Data Transfer

 Source Server         : mysql
 Source Server Type    : MySQL
 Source Server Version : 50717
 Source Host           : localhost:3306
 Source Schema         : movierecsys

 Target Server Type    : MySQL
 Target Server Version : 50717
 File Encoding         : 65001

 Date: 21/03/2021 19:26:28
*/
CREATE database `movierecsys` DEFAULT CHARACTER SET utf8mb4;

use `movierecsys`;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for admin
-- ----------------------------
DROP TABLE IF EXISTS `admin`;
CREATE TABLE `admin`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `pwd` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `is_super` smallint(6) NULL DEFAULT NULL,
  `role_id` int(11) NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name`) USING BTREE,
  INDEX `role_id`(`role_id`) USING BTREE,
  INDEX `ix_admin_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `admin_ibfk_1` FOREIGN KEY (`role_id`) REFERENCES `role` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of admin
-- ----------------------------
INSERT INTO `admin` VALUES (1, 'Henry', 'pbkdf2:sha256:150000$F8OasjVl$4429b52a5bb3c9b373455596e4399ba82faf0bb725e0c6fe842840a51d057af7', 1, 1, '2021-03-20 17:50:59');
INSERT INTO `admin` VALUES (2, 'admin', 'pbkdf2:sha256:150000$xntTNEnP$880a65a03c2d37d039bc1751ef15413c17fc837609ef95e89f2629ce40a97932', 1, 1, '2021-03-20 18:04:25');

-- ----------------------------
-- Table structure for adminlog
-- ----------------------------
DROP TABLE IF EXISTS `adminlog`;
CREATE TABLE `adminlog`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `admin_id` int(11) NULL DEFAULT NULL,
  `ip` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `admin_id`(`admin_id`) USING BTREE,
  INDEX `ix_adminlog_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `adminlog_ibfk_1` FOREIGN KEY (`admin_id`) REFERENCES `admin` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 6 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of adminlog
-- ----------------------------
INSERT INTO `adminlog` VALUES (1, 1, '127.0.0.1', '2021-03-20 17:57:13');
INSERT INTO `adminlog` VALUES (2, 2, '127.0.0.1', '2021-03-20 18:04:48');
INSERT INTO `adminlog` VALUES (3, 1, '127.0.0.1', '2021-03-20 18:25:16');
INSERT INTO `adminlog` VALUES (4, 2, '127.0.0.1', '2021-03-20 18:25:23');
INSERT INTO `adminlog` VALUES (5, 2, '127.0.0.1', '2021-03-20 18:33:02');

-- ----------------------------
-- Table structure for auth
-- ----------------------------
DROP TABLE IF EXISTS `auth`;
CREATE TABLE `auth`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name`) USING BTREE,
  UNIQUE INDEX `url`(`url`) USING BTREE,
  INDEX `ix_auth_addtime`(`addtime`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for comment
-- ----------------------------
DROP TABLE IF EXISTS `comment`;
CREATE TABLE `comment`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL,
  `movie_id` int(11) NULL DEFAULT NULL,
  `user_id` int(11) NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `movie_id`(`movie_id`) USING BTREE,
  INDEX `user_id`(`user_id`) USING BTREE,
  INDEX `ix_comment_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `comment_ibfk_1` FOREIGN KEY (`movie_id`) REFERENCES `movie` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `comment_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for movie
-- ----------------------------
DROP TABLE IF EXISTS `movie`;
CREATE TABLE `movie`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `info` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL,
  `logo` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `star` smallint(6) NULL DEFAULT NULL,
  `playnum` bigint(20) NULL DEFAULT NULL,
  `commentnum` bigint(20) NULL DEFAULT NULL,
  `tag_id` int(11) NULL DEFAULT NULL,
  `area` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `release_time` date NULL DEFAULT NULL,
  `length` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `title`(`title`) USING BTREE,
  UNIQUE INDEX `url`(`url`) USING BTREE,
  UNIQUE INDEX `logo`(`logo`) USING BTREE,
  INDEX `tag_id`(`tag_id`) USING BTREE,
  INDEX `ix_movie_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `movie_ibfk_1` FOREIGN KEY (`tag_id`) REFERENCES `tag` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 11 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of movie
-- ----------------------------
INSERT INTO `movie` VALUES (1, '无依之地 Nomadland (2020)', '2021032118003051c9eca0e216417db3109b8387197079.mp4', '基于Jessica Bruder所著书籍《Nomadland: Surviving America in the Twenty-First Century》，讲述一个60多岁的女人在经济大萧条中失去了一切，她作为一个居住在货车里的现代游牧民，开始了穿越美国西部的旅程。', '202103211800307271888a97234a1b968317f744860f06.webp', 1, 5, 0, 1, '美国', '2021-03-10', '3', '2021-03-21 18:00:31');
INSERT INTO `movie` VALUES (2, '同学麦娜丝', '202103211803311c46959a6f5a4466974691a9860343b8.mp4', '四个高中同学毕业后常常聚在泡沫红茶店刁牌、唬烂三小，但聚会后他们还是要继续面对生活的艰难：电风（郑人硕饰）是保险业务员，领着微薄薪水省吃俭用买了新房，因为女友怀孕将步入婚姻的人生阶段；从事纸扎屋行业又有阴阳眼的闭结（刘冠廷饰），因长期照顾卧病在床的阿嬷错过婚姻，想寻求婚姻介绍所找一个好女人结婚；罐头（纳豆饰）在一次吞药自杀后洗心革面，接下了户政所的工作，因此与他心目中的女神校花麦娜丝重逢；添仔（施名帅饰）是怀才不遇的导演，却在机缘下被政客（陈以文饰）相中开始选举之路…', '20210321180331e86087b581b14227abf965cb214a569f.webp', 1, 3, 0, 3, '美国', '2021-03-17', '4', '2021-03-21 18:03:32');
INSERT INTO `movie` VALUES (3, '送你一朵小红花', '20210321180524d90e631a2c0c4cc1a776a1807afd8ae0.mp4', '　两个抗癌家庭，两组生活轨迹。影片讲述了一个温情的现实故事，思考和直面了每一个普通人都会面临的终极问题——想象死亡随时可能到来，我们唯一要做的就是爱和珍惜。', '2021032118052421d1431ed5b64e9d85d42fc20ece0872.webp', 5, 1, 0, 3, '中国', '2021-03-18', '10', '2021-03-21 18:05:25');
INSERT INTO `movie` VALUES (4, '打开心世界', '20210321180719bfb79df2079447c5896fb12310dccee3.mp4', '电影改编自美国作家吉姆·谢泼德在2017年发行的同名短篇小说。故事背景设定在19世纪的美国东北部，描述一名农妇与新邻居的太太——Taille（凡妮莎·柯比 饰）和Abigail（凯瑟琳·沃特斯顿 饰）在相处过程中，不知不觉地填补了彼此生活中的空白，因而被对方深深吸引，逐渐发展亲密关系的故事。', '20210321180719d957502fba7646248f9988f65e3c4383.webp', 3, 0, 0, 1, '美国', '2021-03-25', '3', '2021-03-21 18:07:20');
INSERT INTO `movie` VALUES (5, '吉祥如意', '20210321180833e1e96fe1edd8488d9a65bbd6652241b0.mp4', '大鹏首次执导现实题材文艺片，故事讲述一位喜剧片导演突发奇想，回到东北农村老家，希望将一家人如何过年拍成一部文艺电影，结果遭遇一系列意外。因拍电影而聚齐的家庭成员们，完成了最后的聚会。', '20210321180833b41a9a44a4774b73bb525ad87b263a5a.webp', 1, 0, 0, 3, '中国', '2021-03-19', '7', '2021-03-21 18:08:33');
INSERT INTO `movie` VALUES (6, '孤味 ', '20210321180929bdeaf2eafabe42d6864631d4ecb01f36.mp4', '　林秀英（陈淑芳 饰）是台南赫赫有名的餐厅老板，在丈夫（龙劭华 饰）无声无息地离家后，便靠著卖虾卷独自抚养三个女儿长大成人，不仅把虾卷从路边摊卖到开餐厅，三个女儿更是成就非凡，大女儿阿青（谢盈萱 饰）是国际舞者，二女儿阿瑜（徐若瑄 饰）在台北当整形医生，小女儿佳佳（孙可芳 饰）则接手自己的餐厅事业。眼看就要苦尽甘来，秀英却在70大寿当天接到丈夫离世的噩耗，在替这位有名无实的丈夫筹办丧礼的同时，竟意外迎来了另一位陪伴丈夫度过晚年的女人，令她不得不再次面对内心埋藏已久的怨怼……', '202103211809290440c4bba24f4e7cb38c29bcac718e8a.webp', 1, 0, 0, 1, '中国', '2021-03-25', '10', '2021-03-21 18:09:29');
INSERT INTO `movie` VALUES (7, '缉魂 ', '202103211810448e60d28e22da4456a410afea0186b480.mp4', '　著名集团董事长王世聪惨死家中，负责此案的检察官梁文超（张震 饰）与妻子刑警阿爆（张钧甯 饰）在调查中得知：死者的儿子王天佑（林晖闵 饰），年轻的新婚妻子李燕（孙安可 饰），多年合伙人万宇凡（李铭顺 饰），甚至死去的前妻唐素贞（张柏嘉 饰），每个人之间都有着错综复杂的关联 。阿爆更是发现隐藏在案件背后的惊人秘密……\r\n　　电影根据江波小说《移魂有术》改编。', '2021032118104408ed304ff6524a269763253cb3f17fd4.webp', 2, 0, 0, 1, '美国', '2021-03-18', '3', '2021-03-21 18:10:44');
INSERT INTO `movie` VALUES (8, '间谍之妻', '20210321181137a5c2325847dd4e3ab8cd037afa7e0701.mp4', '影片讲述1940年的日本神户，正值太平洋战争爆发前夕，福原里子正等待着在外国做贸易生意的丈夫的归来，这时她青梅竹马的儿时玩伴、已经是日军宪兵的津森泰治告诉里子，他丈夫竟然从外国带回一个女人，并且那个女人已经去世。在里子的追问下，才得知丈夫在那里偶然发现了“震惊世界”的事件，并下决心要向全世界曝光这件事，为了丈夫的安全和两人的幸福，里子当然想要阻止丈夫的行动。', '2021032118113706087f77104241cead60875dd3c0eb5c.webp', 1, 0, 0, 1, '日本', '2021-03-10', '6', '2021-03-21 18:11:38');
INSERT INTO `movie` VALUES (9, '亲爱的同志', '202103211812251342e265bf56483686449af5f77bd20c.mp4', '　聚焦曾被苏联政府隐瞒的1962年新切尔卡斯克事件，工人罢工，子弹、血肉、背后事件、一个官员兼母亲的探寻和质疑。', '202103211812258ba54ea4f0bc44b5a221ab652952807a.webp', 1, 1, 0, 1, '美国', '2021-03-25', '3', '2021-03-21 18:12:26');
INSERT INTO `movie` VALUES (10, '新·福音战士剧场版', '20210321181326f18f088924d54d419eac151f1bcb3cf8.mp4', '　　成为“第三次冲击”元凶的碇真嗣，变得像废人一样……真嗣还能振作起来吗？2012年《福音战士新剧场版：Q》之后，这部动画讲述了被封印的故事的后续。', '202103211813264e2b8becc49e4a5da7148832b55634bf.webp', 1, 0, 0, 1, '日本', '2021-03-19', '10', '2021-03-21 18:13:27');

-- ----------------------------
-- Table structure for moviecol
-- ----------------------------
DROP TABLE IF EXISTS `moviecol`;
CREATE TABLE `moviecol`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `movie_id` int(11) NULL DEFAULT NULL,
  `user_id` int(11) NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `movie_id`(`movie_id`) USING BTREE,
  INDEX `user_id`(`user_id`) USING BTREE,
  INDEX `ix_moviecol_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `moviecol_ibfk_1` FOREIGN KEY (`movie_id`) REFERENCES `movie` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `moviecol_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for oplog
-- ----------------------------
DROP TABLE IF EXISTS `oplog`;
CREATE TABLE `oplog`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `admin_id` int(11) NULL DEFAULT NULL,
  `ip` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `reason` varchar(600) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `admin_id`(`admin_id`) USING BTREE,
  INDEX `ix_oplog_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `oplog_ibfk_1` FOREIGN KEY (`admin_id`) REFERENCES `admin` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 17 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of oplog
-- ----------------------------
INSERT INTO `oplog` VALUES (1, 2, '127.0.0.1', '添加标签:动作', '2021-03-20 18:25:42');
INSERT INTO `oplog` VALUES (2, 2, '127.0.0.1', '添加标签:喜剧', '2021-03-20 18:25:47');
INSERT INTO `oplog` VALUES (3, 2, '127.0.0.1', '添加预告:无依之地 Nomadland (2020)', '2021-03-20 18:27:47');
INSERT INTO `oplog` VALUES (4, 2, '127.0.0.1', '添加标签:爱情', '2021-03-20 18:33:56');
INSERT INTO `oplog` VALUES (5, 2, '127.0.0.1', '添加预告:极速车王', '2021-03-21 17:56:34');
INSERT INTO `oplog` VALUES (6, 2, '127.0.0.1', '删除预告:极速车王', '2021-03-21 17:58:35');
INSERT INTO `oplog` VALUES (7, 2, '127.0.0.1', '添加电影:《无依之地 Nomadland (2020)》', '2021-03-21 18:00:31');
INSERT INTO `oplog` VALUES (8, 2, '127.0.0.1', '添加电影:《同学麦娜丝》', '2021-03-21 18:03:32');
INSERT INTO `oplog` VALUES (9, 2, '127.0.0.1', '添加电影:《送你一朵小红花》', '2021-03-21 18:05:25');
INSERT INTO `oplog` VALUES (10, 2, '127.0.0.1', '添加电影:《打开心世界》', '2021-03-21 18:07:20');
INSERT INTO `oplog` VALUES (11, 2, '127.0.0.1', '添加电影:《吉祥如意》', '2021-03-21 18:08:33');
INSERT INTO `oplog` VALUES (12, 2, '127.0.0.1', '添加电影:《孤味 》', '2021-03-21 18:09:29');
INSERT INTO `oplog` VALUES (13, 2, '127.0.0.1', '添加电影:《缉魂 》', '2021-03-21 18:10:45');
INSERT INTO `oplog` VALUES (14, 2, '127.0.0.1', '添加电影:《间谍之妻》', '2021-03-21 18:11:38');
INSERT INTO `oplog` VALUES (15, 2, '127.0.0.1', '添加电影:《亲爱的同志》', '2021-03-21 18:12:26');
INSERT INTO `oplog` VALUES (16, 2, '127.0.0.1', '添加电影:《新·福音战士剧场版》', '2021-03-21 18:13:27');

-- ----------------------------
-- Table structure for preview
-- ----------------------------
DROP TABLE IF EXISTS `preview`;
CREATE TABLE `preview`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `logo` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `title`(`title`) USING BTREE,
  UNIQUE INDEX `logo`(`logo`) USING BTREE,
  INDEX `ix_preview_addtime`(`addtime`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of preview
-- ----------------------------
INSERT INTO `preview` VALUES (1, '无依之地 Nomadland (2020)', '20210320182746068aa11c734845b688a639584d42fb5d.webp', '2021-03-20 18:27:47');

-- ----------------------------
-- Table structure for role
-- ----------------------------
DROP TABLE IF EXISTS `role`;
CREATE TABLE `role`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `auths` varchar(600) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name`) USING BTREE,
  INDEX `ix_role_addtime`(`addtime`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of role
-- ----------------------------
INSERT INTO `role` VALUES (1, '超级管理员', '', '2021-03-20 17:50:59');

-- ----------------------------
-- Table structure for tag
-- ----------------------------
DROP TABLE IF EXISTS `tag`;
CREATE TABLE `tag`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name`) USING BTREE,
  INDEX `ix_tag_addtime`(`addtime`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 4 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of tag
-- ----------------------------
INSERT INTO `tag` VALUES (1, '动作', '2021-03-20 18:25:42');
INSERT INTO `tag` VALUES (2, '喜剧', '2021-03-20 18:25:47');
INSERT INTO `tag` VALUES (3, '爱情', '2021-03-20 18:33:56');

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `pwd` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `email` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `phone` varchar(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `info` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL,
  `face` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  `uuid` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name`) USING BTREE,
  UNIQUE INDEX `email`(`email`) USING BTREE,
  UNIQUE INDEX `phone`(`phone`) USING BTREE,
  UNIQUE INDEX `face`(`face`) USING BTREE,
  UNIQUE INDEX `uuid`(`uuid`) USING BTREE,
  INDEX `ix_user_addtime`(`addtime`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (1, 'user1', 'pbkdf2:sha256:150000$dqwnMB3l$d131b9f68d1af16d0fa1b53cdd526aab0029211f1b70eeea1cebad5e5e25f8aa', 'user2@qq.com', '15145678000', NULL, NULL, '2021-03-20 17:53:16', '5d7384b4ea1a447aa0b9d0ab312122ce');

-- ----------------------------
-- Table structure for userlog
-- ----------------------------
DROP TABLE IF EXISTS `userlog`;
CREATE TABLE `userlog`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NULL DEFAULT NULL,
  `ip` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `addtime` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `user_id`(`user_id`) USING BTREE,
  INDEX `ix_userlog_addtime`(`addtime`) USING BTREE,
  CONSTRAINT `userlog_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 8 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of userlog
-- ----------------------------
INSERT INTO `userlog` VALUES (1, 1, '127.0.0.1', '2021-03-20 17:53:20');
INSERT INTO `userlog` VALUES (2, 1, '127.0.0.1', '2021-03-20 17:55:57');
INSERT INTO `userlog` VALUES (3, 1, '127.0.0.1', '2021-03-20 18:23:06');
INSERT INTO `userlog` VALUES (4, 1, '127.0.0.1', '2021-03-20 18:24:01');
INSERT INTO `userlog` VALUES (5, 1, '127.0.0.1', '2021-03-21 18:13:41');
INSERT INTO `userlog` VALUES (6, 1, '127.0.0.1', '2021-03-21 18:51:28');
INSERT INTO `userlog` VALUES (7, 1, '127.0.0.1', '2021-03-21 19:00:52');

SET FOREIGN_KEY_CHECKS = 1;
