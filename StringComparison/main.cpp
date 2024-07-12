#include <BnchSwt/BenchmarkSuite.hpp>
#include "glaze/glaze.hpp"
#include <concepts>// Required for concepts
#include "../xxHash/xxhash.h"
#include <jsonifier/Index.hpp>
#include <chrishendo/Index.hpp>
//#include "Reflection_Backup.hpp"
//#include "HashMap_Backup.hpp"
//#include "TypeEntities_Backup.hpp"
//#include "Serialize_Impl_Backup.hpp"
//#include "Parse_Impl_Backup.hpp"

struct search_metadata_data {
	std::string since_id_str{};
	std::string next_results{};
	std::string refresh_url{};
	std::string max_id_str{};
	double completed_in{};
	std::string query{};
	int64_t since_id{};
	int64_t count{};
};

struct hashtag {
	std::vector<int64_t> indices{};
	std::string text{};
};

struct large_data {
	std::string resize{};
	hashtag hashTag{};
	int64_t w{};
	int64_t h{};
};

struct sizes_data {
	large_data mediumtestingtesting{};
	large_data smalltestingtesting{};
	large_data thumbtestingtesting{};
	large_data largetestingtesting{};
};

struct media_data {
	std::string expanded_url{};
	std::string display_url{};
	std::string media_url{};
	std::string id_str{};
	std::string type{};
	sizes_data sizes{};
	std::string url{};
};

struct url_data {
	std::vector<int64_t> indices{};
	std::string expanded_url{};
	std::string display_url{};
	std::string url{};
};

struct user_mention {
	std::vector<int64_t> indices{};
	std::string screen_name{};
	std::string id_str{};
	std::string name{};
	int64_t id{};
};

struct status_entities {
	std::optional<std::vector<media_data>> media{};
	std::vector<user_mention> user_mentions{};
	std::vector<std::nullptr_t> symbols{};
	std::vector<hashtag> hashtags{};
	std::vector<url_data> urls{};
	std::string name{};
};

struct metadata_data {
	std::string iso_language_code{};
	std::string result_type{};
};

struct description_data {
	std::vector<url_data> urls{};
};

struct user_entities {
	std::optional<description_data> url{};
	description_data description{};
};

struct twitter_user {
	std::string profile_background_image_url_https{};
	std::optional<std::string> profile_banner_url{};
	std::string profile_sidebar_border_color{};
	std::string profile_background_image_url{};
	std::string profile_sidebar_fill_color{};
	std::optional<std::string> time_zone{};
	std::string profile_background_color{};
	std::string profile_image_url_https{};
	std::optional<int64_t> utc_offset{};
	bool profile_use_background_image{};
	std::optional<std::string> url{};
	std::string profile_text_color{};
	std::string profile_link_color{};
	std::string profile_image_url{};
	bool profile_background_tile{};
	bool is_translation_enabled{};
	bool default_profile_image{};
	bool contributors_enabled{};
	bool follow_request_sent{};
	int64_t favourites_count{};
	std::string description{};
	std::string screen_name{};
	int64_t followers_count{};
	int64_t statuses_count{};
	std::string created_at{};
	user_entities entities{};
	int64_t friends_count{};
	bool default_profile{};
	int64_t listed_count{};
	std::string location{};
	bool user_protected{};
	bool is_translator{};
	std::string id_str{};
	bool notifications{};
	std::string string{};
	std::string name{};
	bool geo_enabled{};
	std::string lang{};
	bool following{};
	bool verified{};
	int64_t id{};
};

struct retweeted_status_data {
	std::optional<std::string> in_reply_to_status_id_str{};
	std::optional<std::string> in_reply_to_user_id_str{};
	std::optional<std::string> in_reply_to_screen_name{};
	std::optional<double> in_reply_to_status_id{};
	std::optional<int64_t> in_reply_to_user_id{};
	std::optional<bool> possibly_sensitive{};
	std::nullptr_t contributors{ nullptr };
	std::nullptr_t coordinates{ nullptr };
	std::nullptr_t place{ nullptr };
	std::nullptr_t geo{ nullptr };
	status_entities entities{};
	int64_t favorite_count{};
	metadata_data metadata{};
	std::string created_at{};
	int64_t retweet_count{};
	std::string source{};
	std::string id_str{};
	twitter_user user{};
	std::string lang{};
	std::string text{};
	bool truncated{};
	bool favorited{};
	bool retweeted{};
	double id{};
};

struct status_data {
	std::optional<retweeted_status_data> retweeted_status{};
	std::optional<std::string> in_reply_to_status_id_str{};
	std::optional<std::string> in_reply_to_user_id_str{};
	std::optional<std::string> in_reply_to_screen_name{};
	std::optional<double> in_reply_to_status_id{};
	std::optional<int64_t> in_reply_to_user_id{};
	std::optional<bool> possibly_sensitive{};
	std::nullptr_t contributors{ nullptr };
	std::nullptr_t coordinates{ nullptr };
	std::nullptr_t place{ nullptr };
	std::nullptr_t geo{ nullptr };
	status_entities entities{};
	int64_t favorite_count{};
	metadata_data metadata{};
	std::string created_at{};
	int64_t retweet_count{};
	std::string source{};
	std::string id_str{};
	twitter_user user{};
	std::string lang{};
	std::string text{};
	bool truncated{};
	bool favorited{};
	bool retweeted{};
	double id{};
};

struct twitter_message {
	search_metadata_data search_metadata{};
	std::vector<status_data> statuses{};
};

struct twitter_message02 {
	search_metadata_data s{};
	std::vector<status_data> d{};
};

struct icon_emoji_data {
	std::optional<std::string> name{};
	std::nullptr_t id{ nullptr };
};

struct permission_overwrite {
	std::string allow{};
	std::string deny{};
	std::string id{};
	int64_t type{};
};

struct channel_data {
	std::vector<permission_overwrite> permission_overwrites{};
	std::optional<std::string> last_message_id{};
	int64_t default_thread_rate_limit_per_user{};
	std::vector<std::nullptr_t> applied_tags{};
	std::vector<std::nullptr_t> recipients{};
	int64_t default_auto_archive_duration{};
	std::nullptr_t rtc_region{ nullptr };
	std::nullptr_t status{ nullptr };
	std::string last_pin_timestamp{};
	std::nullptr_t topic{ nullptr };
	int64_t rate_limit_per_user{};
	icon_emoji_data icon_emoji{};
	int64_t total_message_sent{};
	int64_t video_quality_mode{};
	std::string application_id{};
	std::string permissions{};
	int64_t message_count{};
	std::string parent_id{};
	int64_t member_count{};
	std::string owner_id{};
	std::string guild_id{};
	int64_t user_limit{};
	int64_t position{};
	std::string name{};
	std::string icon{};
	int64_t version{};
	int64_t bitrate{};
	std::string id{};
	int64_t flags{};
	int64_t type{};
	bool managed{};
	bool nsfw{};
};

struct user_data {
	std::nullptr_t avatar_decoration_data{ nullptr };
	std::optional<std::string> display_name{};
	std::optional<std::string> global_name{};
	std::optional<std::string> avatar{};
	std::nullptr_t banner{ nullptr };
	std::nullptr_t locale{ nullptr };
	std::string discriminator{};
	std::string user_name{};
	int64_t accent_color{};
	int64_t premium_type{};
	int64_t public_flags{};
	std::string email{};
	bool mfa_enabled{};
	std::string id{};
	int64_t flags{};
	bool verified{};
};

struct member_data {
	std::nullptr_t communication_disabled_until{ nullptr };
	std::nullptr_t premium_since{ nullptr };
	std::optional<std::string> nick{};
	std::nullptr_t avatar{ nullptr };
	std::vector<std::string> roles{};
	std::string permissions{};
	std::string joined_at{};
	std::string guild_id{};
	user_data user{};
	int64_t flags{};
	bool pending{};
	bool deaf{};
};

struct tags_data {
	std::nullptr_t premium_subscriber{ nullptr };
	std::optional<std::string> bot_id{};
};

struct role_data {
	std::nullptr_t unicode_emoji{ nullptr };
	std::nullptr_t icon{ nullptr };
	std::string permissions{};
	int64_t position{};
	std::string name{};
	bool mentionable{};
	int64_t version{};
	std::string id{};
	tags_data tags{};
	int64_t color{};
	int64_t flags{};
	bool managed{};
	bool hoist{};
};

struct guild_data {
	std::nullptr_t latest_on_boarding_question_id{ nullptr };
	std::vector<std::nullptr_t> guild_scheduled_events{};
	std::nullptr_t safety_alerts_channel_id{ nullptr };
	std::nullptr_t inventory_settings{ nullptr };
	std::vector<std::nullptr_t> voice_states{};
	std::nullptr_t discovery_splash{ nullptr };
	std::nullptr_t vanity_url_code{ nullptr };
	std::nullptr_t application_id{ nullptr };
	std::nullptr_t afk_channel_id{ nullptr };
	int64_t default_message_notifications{};
	int64_t max_stage_video_channel_users{};
	std::string public_updates_channel_id{};
	std::nullptr_t description{ nullptr };
	std::vector<std::nullptr_t> threads{};
	std::vector<channel_data> channels{};
	int64_t premium_subscription_count{};
	int64_t approximate_presence_count{};
	std::vector<std::string> features{};
	std::vector<std::string> stickers{};
	bool premium_progress_bar_enabled{};
	std::vector<member_data> members{};
	std::nullptr_t hub_type{ nullptr };
	int64_t approximate_member_count{};
	int64_t explicit_content_filter{};
	int64_t max_video_channel_users{};
	std::nullptr_t splash{ nullptr };
	std::nullptr_t banner{ nullptr };
	std::string system_channel_id{};
	std::string widget_channel_id{};
	std::string preferred_locale{};
	int64_t system_channel_flags{};
	std::string rules_channel_id{};
	std::vector<role_data> roles{};
	int64_t verification_level{};
	std::string permissions{};
	int64_t max_presences{};
	std::string discovery{};
	std::string joined_at{};
	int64_t member_count{};
	int64_t premium_tier{};
	std::string owner_id{};
	int64_t max_members{};
	int64_t afk_timeout{};
	bool widget_enabled{};
	std::string region{};
	int64_t nsfw_level{};
	int64_t mfa_level{};
	std::string name{};
	std::string icon{};
	bool unavailable{};
	std::string id{};
	int64_t flags{};
	bool large{};
	bool owner{};
	bool nsfw{};
	bool lazy{};
};

struct discord_message {
	std::string t{};
	guild_data d{};
	int64_t op{};
	int64_t s{};
};


template<> struct jsonifier::core<icon_emoji_data> {
	using value_type				 = icon_emoji_data;
	static constexpr auto parseValue = createValue<&value_type::name, &value_type::id>();
};

template<> struct jsonifier::core<permission_overwrite> {
	using value_type				 = permission_overwrite;
	static constexpr auto parseValue = createValue<&value_type::allow, &value_type::deny, &value_type::id, &value_type::type>();
};

template<> struct jsonifier::core<channel_data> {
	using value_type				 = channel_data;
	static constexpr auto parseValue = createValue<&value_type::permission_overwrites, &value_type::last_message_id, &value_type::default_thread_rate_limit_per_user,
		&value_type::applied_tags, &value_type::recipients, &value_type::default_auto_archive_duration, &value_type::rtc_region, &value_type::status,
		&value_type::last_pin_timestamp, &value_type::topic, &value_type::rate_limit_per_user, &value_type::icon_emoji, &value_type::total_message_sent,
		&value_type::video_quality_mode, &value_type::application_id, &value_type::permissions>();
};

template<> struct jsonifier::core<user_data> {
	using value_type				 = user_data;
	static constexpr auto parseValue = createValue<&value_type::avatar_decoration_data, &value_type::display_name, &value_type::global_name, &value_type::avatar,
		&value_type::banner, &value_type::locale, &value_type::discriminator, &value_type::user_name, &value_type::accent_color, &value_type::premium_type,
		&value_type::public_flags, &value_type::email, &value_type::mfa_enabled, &value_type::id, &value_type::flags, &value_type::verified>();
};

template<> struct jsonifier::core<member_data> {
	using value_type = member_data;
	static constexpr auto parseValue =
		createValue<&value_type::communication_disabled_until, &value_type::premium_since, &value_type::nick, &value_type::avatar, &value_type::roles, &value_type::permissions,
			&value_type::joined_at, &value_type::guild_id, &value_type::user, &value_type::flags, &value_type::pending, &value_type::deaf>();
};

template<> struct jsonifier::core<tags_data> {
	using value_type				 = tags_data;
	static constexpr auto parseValue = createValue<&value_type::premium_subscriber, &value_type::bot_id>();
};

template<> struct jsonifier::core<role_data> {
	using value_type				 = role_data;
	static constexpr auto parseValue = createValue<&value_type::unicode_emoji, &value_type::icon, &value_type::permissions, &value_type::position, &value_type::name,
		&value_type::mentionable, &value_type::version, &value_type::id, &value_type::tags, &value_type::color, &value_type::flags, &value_type::managed, &value_type::hoist>();
};

template<> struct jsonifier::core<guild_data> {
	using value_type				 = guild_data;
	static constexpr auto parseValue = createValue<&value_type::latest_on_boarding_question_id, &value_type::guild_scheduled_events, &value_type::safety_alerts_channel_id,
		&value_type::inventory_settings, &value_type::voice_states, &value_type::discovery_splash, &value_type::vanity_url_code, &value_type::application_id,
		&value_type::afk_channel_id, &value_type::default_message_notifications, &value_type::max_stage_video_channel_users, &value_type::public_updates_channel_id,
		&value_type::description, &value_type::threads, &value_type::channels, &value_type::premium_subscription_count, &value_type::approximate_presence_count,
		&value_type::features, &value_type::stickers, &value_type::premium_progress_bar_enabled, &value_type::members, &value_type::hub_type, &value_type::approximate_member_count,
		&value_type::explicit_content_filter, &value_type::max_video_channel_users, &value_type::splash, &value_type::banner, &value_type::system_channel_id,
		&value_type::widget_channel_id, &value_type::preferred_locale, &value_type::system_channel_flags, &value_type::rules_channel_id, &value_type::roles,
		&value_type::verification_level, &value_type::permissions, &value_type::max_presences, &value_type::discovery, &value_type::joined_at, &value_type::member_count,
		&value_type::premium_tier, &value_type::owner_id, &value_type::max_members, &value_type::afk_timeout, &value_type::widget_enabled, &value_type::region,
		&value_type::nsfw_level, &value_type::mfa_level, &value_type::name, &value_type::icon, &value_type::unavailable, &value_type::id, &value_type::flags, &value_type::large,
		&value_type::owner, &value_type::nsfw, &value_type::lazy>();
};

template<> struct jsonifier::core<discord_message> {
	using value_type				 = discord_message;
	static constexpr auto parseValue = createValue<&value_type::t, &value_type::d, &value_type::op, &value_type::s>();
};

template<> struct jsonifier::core<search_metadata_data> {
	using value_type				 = search_metadata_data;
	static constexpr auto parseValue = createValue<&value_type::since_id_str, &value_type::next_results, &value_type::refresh_url, &value_type::max_id_str,
		&value_type::completed_in, &value_type::query, &value_type::since_id, &value_type::count>();
};

template<> struct jsonifier::core<hashtag> {
	using value_type				 = hashtag;
	static constexpr auto parseValue = createValue("indices", &value_type::indices, "text", &value_type::text);
};

template<> struct jsonifier::core<large_data> {
	using value_type				 = large_data;
	static constexpr auto parseValue = createValue<&value_type::resize, &value_type::w, &value_type::h>();
};

template<> struct jsonifier::core<sizes_data> {
	using value_type = sizes_data;
	static constexpr auto parseValue =
		createValue<&value_type::mediumtestingtesting, &value_type::smalltestingtesting, &value_type::thumbtestingtesting, &value_type::largetestingtesting>();
};

template<> struct jsonifier::core<media_data> {
	using value_type = media_data;
	static constexpr auto parseValue =
		createValue<&value_type::expanded_url, &value_type::display_url, &value_type::media_url, &value_type::id_str, &value_type::type, &value_type::sizes, &value_type::url>();
};

template<> struct jsonifier::core<url_data> {
	using value_type				 = url_data;
	static constexpr auto parseValue = createValue<&value_type::indices, &value_type::expanded_url, &value_type::display_url, &value_type::url>();
};

template<> struct jsonifier::core<user_mention> {
	using value_type				 = user_mention;
	static constexpr auto parseValue = createValue<&value_type::indices, &value_type::screen_name, &value_type::id_str, &value_type::name, &value_type::id>();
};

template<> struct jsonifier::core<status_entities> {
	using value_type				 = status_entities;
	static constexpr auto parseValue = createValue<&value_type::media, &value_type::user_mentions, &value_type::symbols, &value_type::hashtags, &value_type::urls>();
};

template<> struct jsonifier::core<metadata_data> {
	using value_type				 = metadata_data;
	static constexpr auto parseValue = createValue<&value_type::iso_language_code, &value_type::result_type>();
};

template<> struct jsonifier::core<description_data> {
	using value_type				 = description_data;
	static constexpr auto parseValue = createValue<&value_type::urls>();
};

template<> struct jsonifier::core<user_entities> {
	using value_type				 = user_entities;
	static constexpr auto parseValue = createValue<&value_type::url, &value_type::description>();
};

template<> struct jsonifier::core<twitter_user> {
	using value_type				 = twitter_user;
	static constexpr auto parseValue = createValue<&value_type::profile_background_image_url_https, &value_type::profile_banner_url, &value_type::profile_background_image_url,
		&value_type::profile_sidebar_border_color, &value_type::profile_sidebar_fill_color, &value_type::time_zone, &value_type::profile_background_color,
		&value_type::profile_image_url_https, &value_type::utc_offset, &value_type::profile_use_background_image, &value_type::url, &value_type::profile_text_color,
		&value_type::profile_link_color, &value_type::profile_image_url, &value_type::profile_background_tile, &value_type::is_translation_enabled,
		&value_type::default_profile_image, &value_type::contributors_enabled, &value_type::follow_request_sent, &value_type::favourites_count, &value_type::description,
		&value_type::screen_name, &value_type::followers_count, &value_type::statuses_count, &value_type::created_at, &value_type::entities, &value_type::friends_count,
		&value_type::default_profile, &value_type::listed_count, &value_type::location, &value_type::user_protected, &value_type::is_translator, &value_type::id_str,
		&value_type::notifications, &value_type::string, &value_type::name, &value_type::geo_enabled, &value_type::lang, &value_type::following, &value_type::verified,
		&value_type::id>();
};

template<> struct jsonifier::core<retweeted_status_data> {
	using value_type				 = retweeted_status_data;
	static constexpr auto parseValue = createValue<&value_type::in_reply_to_status_id_str, &value_type::in_reply_to_user_id_str, &value_type::in_reply_to_screen_name,
		&value_type::in_reply_to_status_id, &value_type::in_reply_to_user_id, &value_type::possibly_sensitive, &value_type::contributors, &value_type::coordinates,
		&value_type::place, &value_type::geo, &value_type::entities, &value_type::favorite_count, &value_type::metadata, &value_type::created_at, &value_type::retweet_count,
		&value_type::source, &value_type::id_str, &value_type::user, &value_type::lang, &value_type::text, &value_type::truncated, &value_type::favorited, &value_type::retweeted,
		&value_type::id>();
};

template<> struct jsonifier::core<status_data> {
	using value_type				 = status_data;
	static constexpr auto parseValue = createValue<&value_type::in_reply_to_status_id_str, &value_type::in_reply_to_user_id_str, &value_type::in_reply_to_screen_name,
		&value_type::in_reply_to_status_id, &value_type::in_reply_to_user_id, &value_type::possibly_sensitive, &value_type::contributors, &value_type::coordinates,
		&value_type::retweeted_status, &value_type::place, &value_type::geo, &value_type::entities, &value_type::favorite_count, &value_type::metadata, &value_type::created_at,
		&value_type::retweet_count, &value_type::source, &value_type::id_str, &value_type::user, &value_type::lang, &value_type::text, &value_type::truncated,
		&value_type::favorited, &value_type::retweeted, &value_type::id>();
};

template<> struct jsonifier::core<twitter_message> {
	using value_type				 = twitter_message;
	static constexpr auto parseValue = createValue<&value_type::search_metadata, &value_type::statuses>();
};

template<> struct jsonifier::core<twitter_message02> {
	using value_type				 = twitter_message02;
	static constexpr auto parseValue = createValue<&value_type::d, &value_type::s>();
};

constexpr uint64_t prime{ 0x00000100000001B3 };
constexpr uint64_t offset{ 0xcbf29ce484222325 };

inline uint64_t rawFnv1A(const char* string, size_t length, uint64_t seed) {
	uint64_t hash = offset ^ seed;
	for (size_t x = 0; x < length; ++x) {
		hash ^= static_cast<uint64_t>(string[x]);
		hash *= prime;
	}
	return hash;
}

struct geometry_data {
	std::vector<std::vector<std::vector<double>>> coordinates{};
	std::string type{};
};

struct properties_data {
	std::string name{};
};

struct feature {
	properties_data properties{};
	geometry_data geometry{};
	std::string type{};
};

struct canada_message {
	std::vector<feature> features{};
	std::string type{};
};

template<> struct jsonifier::core<geometry_data> {
	using value_type				 = geometry_data;
	static constexpr auto parseValue = createValue<&value_type::coordinates, &value_type::type>();
};

template<> struct jsonifier::core<properties_data> {
	using value_type				 = properties_data;
	static constexpr auto parseValue = createValue<&value_type::name>();
};

template<> struct jsonifier::core<feature> {
	using value_type				 = feature;
	static constexpr auto parseValue = createValue<&value_type::properties, &value_type::geometry, &value_type::type>();
};

template<> struct jsonifier::core<canada_message> {
	using value_type				 = canada_message;
	static constexpr auto parseValue = createValue<&value_type::features, &value_type::type>();
};

int main() {
	bnch_swt::file_loader fileNew{ "C:/users/chris/source/repos/jsonifier/json/DiscordData-Minified.json" };
	std::string newString01{};
	std::string newString02{ "{\"resize\":\"232342\",\"w\":3434,\"h\":35445,\"hashTag\":{\"indices\":[3432,234234,564556,6757,12323],\"text\":\"testing_text1234\"}}" };
	jsonifier::jsonifier_core parser{};
	uint32_t newString{};

	chrishendo::key_hasher keyHasher{};
	discord_message  testDataNew{};
	parser.parseJson<jsonifier::parse_options{ .minified = true }>(testDataNew, fileNew.operator std::string&());
	for (auto& value: parser.getErrors()) {
		std::cout << "CURRENT ERROR: " << value.reportError() << std::endl;
	}
	parser.serializeJson(testDataNew, newString02);
	//std::cout << "CURRENT VALUE: " << testDataNew.valueTest << std::endl;
	//std::cout << "CURRENT VALUE: " << newString02 << std::endl;
	/*
	std::cout << "CURRENT DATA: " << newString02 << std::endl;
	auto benchmarkStringLengthRt = [&](auto stringLength) {
		std::string newString02{};
		for (uint64_t y = 0; y < stringLength; ++y) {
			newString02.push_back(static_cast<char>(y));
		}
		return bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
			for (uint64_t x = 0; x < 1024; ++x) {
				auto newString = keyHasher.hashKeyRt(newString02.data(), newString02.size());
				bnch_swt::doNotOptimizeAway(newString);
			}
			return;
		});
	};

	auto benchmarkStringLengthCt = [&](auto stringLength) {
		std::string newString02{};
		for (uint64_t y = 0; y < stringLength; ++y) {
			newString02.push_back(static_cast<char>(y));
		}
		return bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
			for (uint64_t x = 0; x < 1024; ++x) {
				auto newString = keyHasher.hashKeyCt(newString02.data(), newString02.size());
				bnch_swt::doNotOptimizeAway(newString);
			}
			return;
		});
	};
	

	for (uint64_t x = 0; x < 2-1; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	static constexpr jsonifier::string_view newerString{ "TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST" };

	uint64_t newValue{ keyHasher.hashKeyCt(newerString.data(), newerString.size()) };
	uint64_t newValue02{ keyHasher.hashKeyRt(newerString.data(), newerString.size()) };
	uint64_t newValue03{ XXH3_64bits_withSeed(newerString.data(), newerString.size(), 0) };
	//jsonifier_internal::accumulateAvxRt(,)
	std::cout << "CURRENT CT VALUE (REAL): " << newValue << std::endl;
	std::cout << "CURRENT RT VALUE (REAL): " << newValue02 << std::endl;
	uint8_t valuesCt[192]{};
	uint8_t valuesRt[192]{};
	jsonifier_internal::initCustomSecretCt(23, valuesCt);
	jsonifier_internal::initCustomSecretRt(23, valuesRt);
	std::cout << "RT VALUES: " << std::endl;
	for (uint64_t x = 0; x < 192; ++x) {
		std::cout << +valuesRt[x] << ",";
	}
	std::cout  << std::endl;
	std::cout << "CT VALUES: " << std::endl;
	for (uint64_t x = 0; x < 192; ++x) {
		std::cout << +valuesCt[x] << ",";
	}
	std::cout << std::endl;

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 4-2; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 4", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 4", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 4", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 8-4; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 8", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 8", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 8", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 16-8; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 16", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 16", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 16", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 32-16; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 32", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 32", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 32", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 64-32; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 64", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 64", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 64", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 128-64; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 128", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 128", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 128", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 240 - 128; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 240", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 240", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 240", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 384 - 240; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 384", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 384", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 384", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 512 - 384; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 512", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 512", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 512", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 1024 - 512; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1024", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1024", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1024", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	
	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2048", "xhash", "skyblue", 100>([&] {
		for (uint64_t x = 0; x < 2048; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2048", "fn1va", "cyan", 100>([&] {
		for (uint64_t x = 0; x < 2048; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2048", "jsonifier_internal::key_hasher", "steelblue", 100>([&] {
		for (uint64_t x = 0; x < 2048; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::writeJsonData("../../../../HashingTest.json");
	bnch_swt::benchmark_suite<"Hashing Test">::writeMarkdownToFile("../../../../HashingTest.md");
	bnch_swt::benchmark_suite<"Hashing Test">::printResults();
	bnch_swt::benchmark_suite<"Visit Test">::printResults();*/
	return 0;
}
