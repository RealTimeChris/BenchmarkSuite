#if defined(JSONIFIER_CPU_INSTRUCTIONS)
//#undef JSONIFIER_CPU_INSTRUCTIONS
//#define JSONIFIER_CPU_INSTRUCTIONS (JSONIFIER_AVX2 | JSONIFIER_POPCNT)
#endif
#include "UnicodeEmoji.hpp"
#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "glaze/glaze.hpp"
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

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

struct search_metadata_data {
	std::string since_id_str{};
	std::string next_results{};
	std::string refresh_url{};
	std::string max_id_str{};
	double completed_in{};
	std::string query{};
	int64_t since_id{};
	int64_t count{};
	double max_id{};
};

struct hashtag {
	std::vector<int64_t> indices{};
	std::string text{};
};

struct large_data {
	std::string resize{};
	int64_t w{};
	int64_t h{};
};

struct sizes_data {
	large_data medium{};
	large_data small{};
	large_data thumb{};
	large_data large{};
};

struct media_data {
	std::optional<std::string> source_status_id_str{};
	std::optional<double> source_status_id{};
	std::vector<int64_t> indices{};
	std::string media_url_https{};
	std::string expanded_url{};
	std::string display_url{};
	std::string media_url{};
	std::string id_str{};
	std::string type{};
	sizes_data sizes{};
	std::string url{};
	double id{};
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
	bool system{};
	bool bot{};
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
	bool mute{};
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

struct names {};

struct event {
	std::optional<std::string> description{};
	std::vector<int64_t> subTopicIds{};
	std::optional<std::string> logo{};
	std::vector<int64_t> topicIds{};
	std::nullptr_t subjectCode{};
	std::nullptr_t subtitle{};
	std::string name{};
	int64_t id{};
};

struct price {
	int64_t audienceSubCategoryId{};
	int64_t seatCategoryId{};
	int64_t amount{};
};

struct area {
	std::vector<std::nullptr_t> blockIds{};
	int64_t areaId{};
};

struct seat_category {
	std::vector<area> areas{};
	int64_t seatCategoryId{};
};

struct performance {
	std::vector<seat_category> seatCategories{};
	std::optional<std::string> logo{};
	std::nullptr_t seatMapImage{};
	std::vector<price> prices{};
	std::string venueCode{};
	std::nullptr_t name{};
	uint64_t eventId{};
	int64_t start{};
	uint64_t id{};
};

struct venue_names {
	std::string PLEYEL_PLEYEL{};
};

struct citm_catalog_message {
	std::map<std::string, std::string> audienceSubCategoryNames{};
	std::map<std::string, std::vector<uint64_t>> topicSubTopics{};
	std::map<std::string, std::string> seatCategoryNames{};
	std::map<std::string, std::string> subTopicNames{};
	std::map<std::string, std::string> areaNames{};
	std::map<std::string, std::string> topicNames{};
	std::vector<performance> performances{};
	std::map<std::string, event> events{};
	venue_names venueNames{};
	names subjectNames{};
	names blockNames{};
};

template<> struct jsonifier::core<names> {
	using value_type				 = names;
	static constexpr auto parseValue = createValue();
};

template<> struct jsonifier::core<event> {
	using value_type				 = event;
	static constexpr auto parseValue = createValue<&value_type::description, &value_type::subTopicIds, &value_type::logo, &value_type::topicIds, &value_type::subjectCode,
		&value_type::subtitle, &value_type::name, &value_type::id>();
};

template<> struct jsonifier::core<price> {
	using value_type				 = price;
	static constexpr auto parseValue = createValue<&value_type::audienceSubCategoryId, &value_type::seatCategoryId, &value_type::amount>();
};

template<> struct jsonifier::core<area> {
	using value_type				 = area;
	static constexpr auto parseValue = createValue<&value_type::blockIds, &value_type::areaId>();
};

template<> struct jsonifier::core<seat_category> {
	using value_type				 = seat_category;
	static constexpr auto parseValue = createValue<&value_type::areas, &value_type::seatCategoryId>();
};

template<> struct jsonifier::core<performance> {
	using value_type				 = performance;
	static constexpr auto parseValue = createValue<&value_type::seatCategories, &value_type::logo, &value_type::seatMapImage, &value_type::prices, &value_type::venueCode,
		&value_type::name, &value_type::eventId, &value_type::start, &value_type::id>();
};

template<> struct jsonifier::core<venue_names> {
	using value_type				 = venue_names;
	static constexpr auto parseValue = createValue<&value_type::PLEYEL_PLEYEL>();
};

template<> struct jsonifier::core<citm_catalog_message> {
	using value_type = citm_catalog_message;
	static constexpr auto parseValue =
		createValue<&value_type::audienceSubCategoryNames, &value_type::topicSubTopics, &value_type::seatCategoryNames, &value_type::subTopicNames, &value_type::areaNames,
			&value_type::topicNames, &value_type::performances, &value_type::events, &value_type::venueNames, &value_type::subjectNames, &value_type::blockNames>();
};

#if !defined(ASAN_ENABLED)

template<> struct glz::meta<names> {
	using value_type			= names;
	static constexpr auto value = object();
};

template<> struct glz::meta<event> {
	using value_type			= event;
	static constexpr auto value = object("description", &value_type::description, "subTopicIds", &value_type::subTopicIds, "logo", &value_type::logo, "topicIds",
		&value_type::topicIds, "subjectCode", &value_type::subjectCode, "subtitle", &value_type::subtitle, "name", &value_type::name, "id", &value_type::id);
};

template<> struct glz::meta<price> {
	using value_type			= price;
	static constexpr auto value = object("audienceSubCategoryId", &value_type::audienceSubCategoryId, "seatCategoryId", &value_type::seatCategoryId, "amount", &value_type::amount);
};

template<> struct glz::meta<area> {
	using value_type			= area;
	static constexpr auto value = object("blockIds", &value_type::blockIds, "areaId", &value_type::areaId);
};

template<> struct glz::meta<seat_category> {
	using value_type			= seat_category;
	static constexpr auto value = object("areas", &value_type::areas, "seatCategoryId", &value_type::seatCategoryId);
};

template<> struct glz::meta<performance> {
	using value_type			= performance;
	static constexpr auto value = object("seatCategories", &value_type::seatCategories, "logo", &value_type::logo, "seatMapImage", &value_type::seatMapImage, "prices",
		&value_type::prices, "venueCode", &value_type::venueCode, "name", &value_type::name, "eventId", &value_type::eventId, "start", &value_type::start, "id", &value_type::id);
};

template<> struct glz::meta<venue_names> {
	using value_type			= venue_names;
	static constexpr auto value = object("PLEYEL_PLEYEL", &value_type::PLEYEL_PLEYEL);
};

template<> struct glz::meta<citm_catalog_message> {
	using value_type			= citm_catalog_message;
	static constexpr auto value = object("audienceSubCategoryNames", &value_type::audienceSubCategoryNames, "topicSubTopics", &value_type::topicSubTopics, "seatCategoryNames",
		&value_type::seatCategoryNames, "subTopicNames", &value_type::subTopicNames, "areaNames", &value_type::areaNames, "topicNames", &value_type::topicNames, "performances",
		&value_type::performances, "events", &value_type::events, "venueNames", &value_type::venueNames, "subjectNames", &value_type::subjectNames, "blockNames",
		&value_type::blockNames);
};
#endif

template<> struct jsonifier::core<geometry_data> {
	using value_type				 = geometry_data;
	static constexpr auto parseValue = createValue("coordinates", &value_type::coordinates, "type", &value_type::type);
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

#if !defined(ASAN_ENABLED)

template<> struct glz::meta<geometry_data> {
	using value_type			= geometry_data;
	static constexpr auto value = object("coordinates", &value_type::coordinates, "type", &value_type::type);
};

template<> struct glz::meta<properties_data> {
	using value_type			= properties_data;
	static constexpr auto value = object("name", &value_type::name);
};

template<> struct glz::meta<feature> {
	using value_type			= feature;
	static constexpr auto value = object("properties", &value_type::properties, "geometry", &value_type::geometry, "type", &value_type::type);
};

template<> struct glz::meta<canada_message> {
	using value_type			= canada_message;
	static constexpr auto value = object("features", &value_type::features, "type", &value_type::type);
};

#endif

template<> struct jsonifier::core<search_metadata_data> {
	using value_type				 = search_metadata_data;
	static constexpr auto parseValue = createValue<&value_type::since_id_str, &value_type::next_results, &value_type::refresh_url, &value_type::max_id_str,
		&value_type::completed_in, &value_type::query, &value_type::since_id, &value_type::count, &value_type::max_id>();
};

template<> struct jsonifier::core<hashtag> {
	using value_type				 = hashtag;
	static constexpr auto parseValue = createValue<&value_type::indices, &value_type::text>();
};

template<> struct jsonifier::core<large_data> {
	using value_type				 = large_data;
	static constexpr auto parseValue = createValue<&value_type::resize, &value_type::w, &value_type::h>();
};

template<> struct jsonifier::core<sizes_data> {
	using value_type				 = sizes_data;
	static constexpr auto parseValue = createValue<&value_type::medium, &value_type::small, &value_type::thumb, &value_type::large>();
};

template<> struct jsonifier::core<media_data> {
	using value_type = media_data;
	static constexpr auto parseValue =
		createValue<&value_type::source_status_id_str, &value_type::source_status_id, &value_type::indices, &value_type::media_url_https, &value_type::expanded_url,
			&value_type::display_url, &value_type::media_url, &value_type::id_str, &value_type::type, &value_type::sizes, &value_type::url, &value_type::id>();
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

template<> struct glz::meta<search_metadata_data> {
	using value_type = search_metadata_data;
	static constexpr auto value =
		object("since_id_str", &value_type::since_id_str, "next_results", &value_type::next_results, "refresh_url", &value_type::refresh_url, "max_id_str", &value_type::max_id_str,
			"completed_in", &value_type::completed_in, "query", &value_type::query, "since_id", &value_type::since_id, "count", &value_type::count, "max_id", &value_type::max_id);
};

template<> struct glz::meta<hashtag> {
	using value_type			= hashtag;
	static constexpr auto value = object("indices", &value_type::indices, "text", &value_type::text);
};

template<> struct glz::meta<large_data> {
	using value_type			= large_data;
	static constexpr auto value = object("resize", &value_type::resize, "w", &value_type::w, "h", &value_type::h);
};

template<> struct glz::meta<sizes_data> {
	using value_type			= sizes_data;
	static constexpr auto value = object("medium", &value_type::medium, "small", &value_type::small, "thumb", &value_type::thumb, "large", &value_type::large);
};

template<> struct glz::meta<media_data> {
	using value_type			= media_data;
	static constexpr auto value = object("source_status_id_str", &value_type::source_status_id_str, "source_status_id", &value_type::source_status_id, "indices",
		&value_type::indices, "media_url_https", &value_type::media_url_https, "expanded_url", &value_type::expanded_url, "display_url", &value_type::display_url, "media_url",
		&value_type::media_url, "id_str", &value_type::id_str, "type", &value_type::type, "sizes", &value_type::sizes, "url", &value_type::url, "id", &value_type::id);
};

template<> struct glz::meta<url_data> {
	using value_type = url_data;
	static constexpr auto value =
		object("indices", &value_type::indices, "expanded_url", &value_type::expanded_url, "display_url", &value_type::display_url, "url", &value_type::url);
};

template<> struct glz::meta<user_mention> {
	using value_type = user_mention;
	static constexpr auto value =
		object("indices", &value_type::indices, "screen_name", &value_type::screen_name, "id_str", &value_type::id_str, "name", &value_type::name, "id", &value_type::id);
};

template<> struct glz::meta<status_entities> {
	using value_type			= status_entities;
	static constexpr auto value = object("media", &value_type::media, "user_mentions", &value_type::user_mentions, "symbols", &value_type::symbols, "hashtags",
		&value_type::hashtags, "urls", &value_type::urls);
};

template<> struct glz::meta<metadata_data> {
	using value_type			= metadata_data;
	static constexpr auto value = object("iso_language_code", &value_type::iso_language_code, "result_type", &value_type::result_type);
};

template<> struct glz::meta<description_data> {
	using value_type			= description_data;
	static constexpr auto value = object("urls", &value_type::urls);
};

template<> struct glz::meta<user_entities> {
	using value_type			= user_entities;
	static constexpr auto value = object("url", &value_type::url, "description", &value_type::description);
};

template<> struct glz::meta<twitter_user> {
	using value_type			= twitter_user;
	static constexpr auto value = object("profile_background_image_url_https", &value_type::profile_background_image_url_https, "profile_banner_url",
		&value_type::profile_banner_url, "profile_background_image_url", &value_type::profile_banner_url, "profile_sidebar_border_color", &value_type::profile_sidebar_border_color,
		"profile_sidebar_fill_color", &value_type::profile_sidebar_fill_color, "time_zone", &value_type::time_zone, "profile_background_color",
		&value_type::profile_background_color, "profile_image_url_https", &value_type::profile_image_url_https, "utc_offset", &value_type::utc_offset,
		"profile_use_background_image", &value_type::profile_use_background_image, "url", &value_type::url, "profile_text_color", &value_type::profile_text_color,
		"profile_link_color", &value_type::profile_link_color, "profile_image_url", &value_type::profile_image_url, "profile_background_tile", &value_type::profile_background_tile,
		"is_translation_enabled", &value_type::is_translation_enabled, "default_profile_image", &value_type::default_profile_image, "contributors_enabled",
		&value_type::contributors_enabled, "follow_request_sent", &value_type::follow_request_sent, "favourites_count", &value_type::favourites_count, "description",
		&value_type::description, "screen_name", &value_type::screen_name, "followers_count", &value_type::followers_count, "statuses_count", &value_type::statuses_count,
		"created_at", &value_type::created_at, "entities", &value_type::entities, "friends_count", &value_type::friends_count, "default_profile", &value_type::default_profile,
		"listed_count", &value_type::listed_count, "location", &value_type::location, "user_protected", &value_type::user_protected, "is_translator", &value_type::is_translator,
		"id_str", &value_type::id_str, "notifications", &value_type::notifications, "string", &value_type::string, "name", &value_type::name, "geo_enabled",
		&value_type::geo_enabled, "lang", &value_type::lang, "following", &value_type::following, "verified", &value_type::verified, "id", &value_type::id);
};

template<> struct glz::meta<retweeted_status_data> {
	using value_type			= retweeted_status_data;
	static constexpr auto value = object("in_reply_to_status_id_str", &value_type::in_reply_to_status_id_str, "in_reply_to_user_id_str", &value_type::in_reply_to_user_id_str,
		"in_reply_to_screen_name", &value_type::in_reply_to_screen_name, "in_reply_to_status_id", &value_type::in_reply_to_status_id, "in_reply_to_user_id",
		&value_type::in_reply_to_user_id, "possibly_sensitive", &value_type::possibly_sensitive, "contributors", &value_type::contributors, "coordinates", &value_type::coordinates,
		"place", &value_type::place, "geo", &value_type::geo, "entities", &value_type::entities, "favorite_count", &value_type::favorite_count, "metadata", &value_type::metadata,
		"created_at", &value_type::created_at, "retweet_count", &value_type::retweet_count, "source", &value_type::source, "id_str", &value_type::id_str, "user", &value_type::user,
		"lang", &value_type::lang, "text", &value_type::text, "truncated", &value_type::truncated, "favorited", &value_type::favorited, "retweeted", &value_type::retweeted, "id",
		&value_type::id);
};

template<> struct glz::meta<status_data> {
	using value_type			= status_data;
	static constexpr auto value = object("in_reply_to_status_id_str", &value_type::in_reply_to_status_id_str, "in_reply_to_user_id_str", &value_type::in_reply_to_user_id_str,
		"in_reply_to_screen_name", &value_type::in_reply_to_screen_name, "in_reply_to_status_id", &value_type::in_reply_to_status_id, "in_reply_to_user_id",
		&value_type::in_reply_to_user_id, "possibly_sensitive", &value_type::possibly_sensitive, "contributors", &value_type::contributors, "coordinates", &value_type::coordinates,
		"place", &value_type::place, "geo", &value_type::geo, "entities", &value_type::entities, "favorite_count", &value_type::favorite_count, "metadata", &value_type::metadata,
		"created_at", &value_type::created_at, "retweeted_status", &value_type::retweeted_status, "retweet_count", &value_type::retweet_count, "source", &value_type::source,
		"id_str", &value_type::id_str, "user", &value_type::user, "lang", &value_type::lang, "text", &value_type::text, "truncated", &value_type::truncated, "favorited",
		&value_type::favorited, "retweeted", &value_type::retweeted, "id", &value_type::id);
};

template<> struct glz::meta<twitter_message> {
	using value_type			= twitter_message;
	static constexpr auto value = object("search_metadata", &value_type::search_metadata, "statuses", &value_type::statuses);
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
		&value_type::applied_tags, &value_type::recipients, &value_type::default_auto_archive_duration, &value_type::status, &value_type::last_pin_timestamp, &value_type::topic,
		&value_type::rate_limit_per_user, &value_type::icon_emoji, &value_type::total_message_sent, &value_type::video_quality_mode, &value_type::application_id,
		&value_type::permissions, &value_type::message_count, &value_type::parent_id, &value_type::member_count, &value_type::owner_id, &value_type::guild_id,
		&value_type::user_limit, &value_type::position, &value_type::name, &value_type::icon, &value_type::version, &value_type::bitrate, &value_type::id, &value_type::flags,
		&value_type::type, &value_type::managed, &value_type::nsfw>();
};

template<> struct jsonifier::core<user_data> {
	using value_type = user_data;
	static constexpr auto parseValue =
		createValue<&value_type::avatar_decoration_data, &value_type::display_name, &value_type::global_name, &value_type::avatar, &value_type::banner, &value_type::locale,
			&value_type::discriminator, &value_type::user_name, &value_type::accent_color, &value_type::premium_type, &value_type::public_flags, &value_type::email,
			&value_type::mfa_enabled, &value_type::id, &value_type::flags, &value_type::verified, &value_type::system, &value_type::bot>();
};

template<> struct jsonifier::core<member_data> {
	using value_type = member_data;
	static constexpr auto parseValue =
		createValue<&value_type::communication_disabled_until, &value_type::premium_since, &value_type::nick, &value_type::avatar, &value_type::roles, &value_type::permissions,
			&value_type::joined_at, &value_type::guild_id, &value_type::user, &value_type::flags, &value_type::pending, &value_type::deaf, &value_type::mute>();
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

int main() {
	bnch_swt::file_loader file{ JSON_BASE_PATH + std::string{ "/CanadaData-Prettified.json" } };
	canada_message dataNew{};
	jsonifier::jsonifier_core parser{};
	parser.parseJson(dataNew, file.operator jsonifier::string&());
	for (auto& value: parser.getErrors()) {
		std::cout << "ERROR: " << value << std::endl;
	}
	
	return 0;
}