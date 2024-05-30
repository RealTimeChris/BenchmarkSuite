#include <BnchSwt/BenchmarkSuite.hpp>
#include "glaze/glaze.hpp"
#include <concepts>// Required for concepts
#include "../xxHash/xxhash.h"
#include <jsonifier/Index.hpp>
#include <chrishendo/Index.hpp>
/*

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

template<> struct glz::meta<icon_emoji_data> {
	using value_type			= icon_emoji_data;
	static constexpr auto value = object("name", &value_type::name, "id", &value_type::id);
};

template<> struct glz::meta<permission_overwrite> {
	using value_type			= permission_overwrite;
	static constexpr auto value = object("allow", &value_type::allow, "deny", &value_type::deny, "id", &value_type::id, "type", &value_type::type);
};

template<> struct glz::meta<channel_data> {
	using value_type			= channel_data;
	static constexpr auto value = object("permission_overwrites", &value_type::permission_overwrites, "last_message_id", &value_type::last_message_id,
		"default_thread_rate_limit_per_user", &value_type::default_thread_rate_limit_per_user, "applied_tags", &value_type::applied_tags, "recipients", &value_type::recipients,
		"default_auto_archive_duration", &value_type::default_auto_archive_duration, "rtc_region", &value_type::rtc_region, "status", &value_type::status, "last_pin_timestamp",
		&value_type::last_pin_timestamp, "topic", &value_type::topic, "rate_limit_per_user", &value_type::rate_limit_per_user, "icon_emoji", &value_type::icon_emoji,
		"total_message_sent", &value_type::total_message_sent, "video_quality_mode", &value_type::video_quality_mode, "application_id", &value_type::application_id, "permissions",
		&value_type::permissions, "message_count", &value_type::message_count, "parent_id", &value_type::parent_id, "member_count", &value_type::member_count, "owner_id",
		&value_type::owner_id, "guild_id", &value_type::guild_id, "user_limit", &value_type::user_limit, "position", &value_type::position, "name", &value_type::name, "icon",
		&value_type::icon, "version", &value_type::version, "bitrate", &value_type::bitrate, "id", &value_type::id, "flags", &value_type::flags, "type", &value_type::type,
		"managed", &value_type::managed, "nsfw", &value_type::nsfw);
};

template<> struct glz::meta<user_data> {
	using value_type			= user_data;
	static constexpr auto value = object("avatar_decoration_data", &value_type::avatar_decoration_data, "display_name", &value_type::display_name, "global_name",
		&value_type::global_name, "avatar", &value_type::avatar, "banner", &value_type::banner, "locale", &value_type::locale, "discriminator", &value_type::discriminator,
		"user_name", &value_type::user_name, "accent_color", &value_type::accent_color, "premium_type", &value_type::premium_type, "public_flags", &value_type::public_flags,
		"email", &value_type::email, "mfa_enabled", &value_type::mfa_enabled, "id", &value_type::id, "flags", &value_type::flags, "verified", &value_type::verified);
};

template<> struct glz::meta<member_data> {
	using value_type			= member_data;
	static constexpr auto value = object("communication_disabled_until", &value_type::communication_disabled_until, "premium_since", &value_type::premium_since, "nick",
		&value_type::nick, "avatar", &value_type::avatar, "roles", &value_type::roles, "permissions", &value_type::permissions, "joined_at", &value_type::joined_at, "guild_id",
		&value_type::guild_id, "user", &value_type::user, "flags", &value_type::flags, "pending", &value_type::pending, "deaf", &value_type::deaf);
};

template<> struct glz::meta<tags_data> {
	using value_type			= tags_data;
	static constexpr auto value = object("premium_subscriber", &value_type::premium_subscriber, "bot_id", &value_type::bot_id);
};

template<> struct glz::meta<role_data> {
	using value_type			= role_data;
	static constexpr auto value = object("unicode_emoji", &value_type::unicode_emoji, "icon", &value_type::icon, "permissions", &value_type::permissions, "position",
		&value_type::position, "name", &value_type::name, "mentionable", &value_type::mentionable, "version", &value_type::version, "id", &value_type::id, "tags",
		&value_type::tags, "color", &value_type::color, "flags", &value_type::flags, "managed", &value_type::managed, "hoist", &value_type::hoist);
};

template<> struct glz::meta<guild_data> {
	using value_type			= guild_data;
	static constexpr auto value = object("latest_on_boarding_question_id", &value_type::latest_on_boarding_question_id, "guild_scheduled_events",
		&value_type::guild_scheduled_events, "safety_alerts_channel_id", &value_type::safety_alerts_channel_id, "inventory_settings", &value_type::inventory_settings,
		"voice_states", &value_type::voice_states, "discovery_splash", &value_type::discovery_splash, "vanity_url_code", &value_type::vanity_url_code, "application_id",
		&value_type::application_id, "afk_channel_id", &value_type::afk_channel_id, "default_message_notifications", &value_type::default_message_notifications,
		"max_stage_video_channel_users", &value_type::max_stage_video_channel_users, "public_updates_channel_id", &value_type::public_updates_channel_id, "description",
		&value_type::description, "threads", &value_type::threads, "channels", &value_type::channels, "premium_subscription_count", &value_type::premium_subscription_count,
		"approximate_presence_count", &value_type::approximate_presence_count, "features", &value_type::features, "stickers", &value_type::stickers, "premium_progress_bar_enabled",
		&value_type::premium_progress_bar_enabled, "members", &value_type::members, "hub_type", &value_type::hub_type, "approximate_member_count",
		&value_type::approximate_member_count, "explicit_content_filter", &value_type::explicit_content_filter, "max_video_channel_users", &value_type::max_video_channel_users,
		"splash", &value_type::splash, "banner", &value_type::banner, "system_channel_id", &value_type::system_channel_id, "widget_channel_id", &value_type::widget_channel_id,
		"preferred_locale", &value_type::preferred_locale, "system_channel_flags", &value_type::system_channel_flags, "rules_channel_id", &value_type::rules_channel_id, "roles",
		&value_type::roles, "verification_level", &value_type::verification_level, "permissions", &value_type::permissions, "max_presences", &value_type::max_presences,
		"discovery", &value_type::discovery, "joined_at", &value_type::joined_at, "member_count", &value_type::member_count, "premium_tier", &value_type::premium_tier, "owner_id",
		&value_type::owner_id, "max_members", &value_type::max_members, "afk_timeout", &value_type::afk_timeout, "widget_enabled", &value_type::widget_enabled, "region",
		&value_type::region, "nsfw_level", &value_type::nsfw_level, "mfa_level", &value_type::mfa_level, "name", &value_type::name, "icon", &value_type::icon, "unavailable",
		&value_type::unavailable, "id", &value_type::id, "flags", &value_type::flags, "large", &value_type::large, "owner", &value_type::owner, "nsfw", &value_type::nsfw, "lazy");
};

template<> struct jsonifier::core<search_metadata_data> {
	using value_type				 = search_metadata_data;
	static constexpr auto parseValue = createValue<&value_type::since_id_str, &value_type::next_results, &value_type::refresh_url, &value_type::max_id_str,
		&value_type::completed_in, &value_type::query, &value_type::since_id, &value_type::count>();
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

template<> struct glz::meta<search_metadata_data> {
	using value_type = search_metadata_data;
	static constexpr auto value =
		object("since_id_str", &value_type::since_id_str, "next_results", &value_type::next_results, "refresh_url", &value_type::refresh_url, "max_id_str", &value_type::max_id_str,
			"completed_in", &value_type::completed_in, "query", &value_type::query, "since_id", &value_type::since_id, "count", &value_type::count);
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
	static constexpr auto value = object("mediumtestingtesting", &value_type::mediumtestingtesting, "smalltestingtesting", &value_type::smalltestingtesting, "thumbtestingtesting",
		&value_type::thumbtestingtesting, "largetestingtesting", &value_type::largetestingtesting);
};

template<> struct glz::meta<media_data> {
	using value_type			= media_data;
	static constexpr auto value = object("expanded_url", &value_type::expanded_url, "display_url", &value_type::display_url, "media_url", &value_type::media_url, "id_str",
		&value_type::id_str, "type", &value_type::type, "sizes", &value_type::sizes, "url", &value_type::url);
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

#include <random>

inline std::random_device randomEngine{};
inline std::mt19937_64 gen{ randomEngine() };

constexpr jsonifier::string_view charset{ "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\"\\\r\b\f\t\n" };

uint64_t randomizeNumberUniform(uint64_t lowEnd, uint64_t highEnd) {
	std::uniform_int_distribution<uint64_t> dis(lowEnd, highEnd);
	return dis(randomEngine);
}

std::string generateString() {
	uint64_t length = randomizeNumberUniform(64, 128);
	std::string result{};
	for (int32_t x = 0; x < length; ++x) {
		result += charset[randomizeNumberUniform(0, charset.size())];
	}
	return result;
}

template<uint64_t length> struct convert_length_to_int {
	static_assert(length <= 8, "Sorry, but that string is too long!");
	using type = std::conditional_t<length == 1, uint8_t,
		std::conditional_t<length <= 2, uint16_t, std::conditional_t<length <= 4, uint32_t, std::conditional_t<length <= 8, uint64_t, void>>>>;
};

template<uint64_t length> using convert_length_to_int_t = typename convert_length_to_int<length>::type;

template<bnch_swt::string_literal string> constexpr convert_length_to_int_t<string.size()> getStringAsInt() {
	const char* stringNew = string.data();
	convert_length_to_int_t<string.size()> returnValue{};
	for (uint64_t x = 0; x < string.size(); ++x) {
		returnValue |= static_cast<convert_length_to_int_t<string.size()>>(stringNew[x]) << x * 8;
	}
	return returnValue;
}

template<bnch_swt::string_literal string> JSONIFIER_INLINE bool compareStringAsInt(const char* iter) {
	constexpr auto newString{ getStringAsInt<string>() };
	convert_length_to_int_t<string.size()> newString02{};
	std::memcpy(&newString02, iter, string.size());
	return newString == newString02;
}

template<typename iterator_type, jsonifier::concepts::bool_t bool_type> JSONIFIER_INLINE bool parseBool(bool_type& value, iterator_type&& iter) {
	if (compareStringAsInt<"true">(iter)) {
		value = true;
		iter += 4;
		return true;
	} else if (compareStringAsInt<"false">(iter)) {
		value = false;
		iter += 5;
		return true;
	} else {
		return false;
	}
}

template<typename iterator_type, jsonifier::concepts::bool_t bool_type> JSONIFIER_INLINE bool parseBool02(bool_type& value, iterator_type&& iter) {
	uint64_t c{};
	// Note that because our buffer must be null terminated, we can read one more index without checking:
	std::memcpy(&c, iter, 5);
	constexpr uint64_t u_true  = 0b00000000'00000000'00000000'00000000'01100101'01110101'01110010'01110100;
	constexpr uint64_t u_false = 0b00000000'00000000'00000000'01100101'01110011'01101100'01100001'01100110;
	// We have to wipe the 5th character for true testing
	if ((c & 0xFF'FF'FF'00'FF'FF'FF'FF) == u_true) {
		value = true;
		iter += 4;
	} else {
		if (c != u_false) [[unlikely]] {
			return false;
		}
		value = false;
		iter += 5;
	}
	return true;
}

template<jsonifier::concepts::json_structural_iterator_t iterator_type, jsonifier::concepts::bool_t bool_type>
JSONIFIER_INLINE bool parseBool(bool_type& value, iterator_type&& iter) {
	if (compareStringAsInt<"true">(iter)) {
		value = true;
		++iter;
		return true;
	} else if (compareStringAsInt<"false">(iter)) {
		value = false;
		++iter;
		return true;
	} else {
		return false;
	}
}

template<typename iterator_type> JSONIFIER_INLINE bool parseNull(iterator_type&& iter) {
	if (compareStringAsInt<"null">(iter)) {
		iter += 4;
		return true;
	} else {
		return false;
	}
}

template<typename iterator_type> JSONIFIER_INLINE bool parseNull02(iterator_type&& iter) {
	constexpr bnch_swt::string_literal str{ "null" };
	if (!glz::compare<str.size()>(iter, str.values)) [[unlikely]] {
		return false;
	} else [[likely]] {
		iter += str.size();
		return true;
	}
}

template<typename value_type, uint64_t size> constexpr auto convertRawToArray(value_type values[size]) {
	std::array<value_type, size> returnValues{};
	for (uint64_t x = 0; x < size; ++x) {
		returnValues[x] = values[x];
	}
	return returnValues;
}

template<typename value_type, std::size_t N> constexpr auto convertRawToArray(value_type (&values)[N]) {
	std::array<std::remove_const_t<value_type>, N> result{};
	for (std::size_t x = 0; x < N; ++x) {
		result[x] = values[x];
	}
	return result;
}

struct testStruct {
	jsonifier::raw_json_data resize{};
	jsonifier::raw_json_data w{};
	jsonifier::raw_json_data h{};
};

template<> struct jsonifier::core<testStruct> {
	using value_type				 = testStruct;
	static constexpr auto parseValue = createValue<&value_type::h, &value_type::w, &value_type::resize>();
};

template<uint64_t shiftNew> JSONIFIER_INLINE constexpr uint64_t xorShift6401(uint64_t v64) {
	constexpr auto shift{ shiftNew };
	return v64 ^ (v64 >> shift);
}

template<uint64_t shiftNew> JSONIFIER_INLINE constexpr uint64_t xorShift6402(uint64_t v64) {
	return v64 ^ (v64 >> shiftNew);
}

constexpr uint64_t xorShift64(uint64_t v64, uint64_t shiftNew) {
	return v64 ^ (v64 >> shiftNew);
}

template<std::size_t N, class Func> constexpr void for_each_short_circuit(Func&& f) {
	[&]<std::size_t... I>(std::index_sequence<I...>) constexpr {
		(f(std::integral_constant<std::size_t, I>{}) || ...);
	}(std::make_index_sequence<N>{});
}

void read_json_visitor(auto&& value, auto&& variant) noexcept {
	constexpr auto variant_size = std::variant_size_v<std::decay_t<decltype(variant)>>;
	for_each_short_circuit<variant_size>([&](auto I) {
		if (I == variant.index()) {
			auto newString = jsonifier_internal::getMember(value, std::get<I>(variant));
			bnch_swt::doNotOptimizeAway(newString);
			return true;
		}
		return false;
	});
}


*/

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

int32_t main() {
	std::string newString01{};
	for (uint64_t x = 0; x < 1; ++x) {
		newString01.push_back(static_cast<char>(x));
	}
	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	chrishendo::key_hasher keyHasher{};

	auto benchmarkStringLengthRt = [&](auto stringLength) {
		std::string newString02{};
		for (uint64_t y = 0; y < stringLength; ++y) {
			newString02.push_back(static_cast<char>(y));
		}
		return bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
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
		return bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
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

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 4-2; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 4", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 4", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 4", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 8-4; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 8", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 8", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 8", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 16-8; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 16", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 16", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 16", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 32-16; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 32", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 32", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 32", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 64-32; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 64", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 64", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 64", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 128-64; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 128", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 128", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 128", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 240 - 128; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 240", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 240", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 240", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 384 - 240; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 384", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 384", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 384", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 512 - 384; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 512", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 512", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 512", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	for (uint64_t x = 0; x < 1024 - 512; ++x) {
		newString01.push_back(static_cast<char>(x));
	}

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1024", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1024", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 1024", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 1024; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	
	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2048", "xhash", "skyblue", 5000>([&] {
		for (uint64_t x = 0; x < 2048; ++x) {
			auto newString = XXH3_64bits(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2048", "fn1va", "cyan", 5000>([&] {
		for (uint64_t x = 0; x < 2048; ++x) {
			auto newString = rawFnv1A(newString01.data(), newString01.size(), 0);
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::benchmark<"Hashing Test: Length 2048", "jsonifier_internal::key_hasher", "steelblue", 5000>([&] {
		for (uint64_t x = 0; x < 2048; ++x) {
			auto newString = keyHasher.hashKeyRt(newString01.data(), newString01.size());
			bnch_swt::doNotOptimizeAway(newString);
		}
		return;
	});

	bnch_swt::benchmark_suite<"Hashing Test">::writeJsonData("../../../../HashingTest.json");
	bnch_swt::benchmark_suite<"Hashing Test">::writeMarkdownToFile("../../../../HashingTest.md");
	bnch_swt::benchmark_suite<"Hashing Test">::printResults();
	return 0;
}
